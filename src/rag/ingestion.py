from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def load_pdfs(pdf_dir: str, min_content_length: int = 20) -> list[Document]:
    """
    Load semua PDF di pdf_dir, tambahkan metadata 'source_file'
    (nama file, jadi identitas UU/PP-nya) ke tiap Document.

    - Case-insensitive untuk ekstensi .pdf/.PDF.
    - Urutan file di-sort supaya log/debug konsisten antar run.
    - Warning (bukan auto-fix) untuk halaman dengan konten nyaris kosong
      (indikasi hasil scan/gambar tanpa OCR) -- supaya kamu sadar dari awal
      kalau ada bagian dokumen yang bakal invisible di retrieval nanti.

    Return: list gabungan semua Document dari semua file PDF di pdf_dir.
    """
    pdf_paths = sorted(
        {*Path(pdf_dir).glob("*.pdf"), *Path(pdf_dir).glob("*.PDF")},
        key=lambda p: p.name,
    )

    documents: list[Document] = []
    thin_content_count = 0

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            # normalize source path jadi nama file bersih
            doc.metadata["source_file"] = pdf_path.name

            if len(doc.page_content.strip()) < min_content_length:
                thin_content_count += 1
                print(
                    f"[WARNING] Konten nyaris kosong (<{min_content_length} char) "
                    f"di {pdf_path.name}, halaman {doc.metadata.get('page', '?')}. "
                    f"Kemungkinan hasil scan/gambar tanpa OCR -- cek manual, "
                    f"karena halaman ini efektif invisible untuk retrieval."
                )

        documents.extend(docs)

    if thin_content_count > 0:
        print(
            f"\n[SUMMARY] Total {thin_content_count} halaman dengan konten "
            f"nyaris kosong dari {len(pdf_paths)} file PDF. "
            f"Pertimbangkan OCR manual kalau jumlah ini signifikan."
        )

    return documents


def validate_pdf_count(documents: list[Document], expected_files: int = 4) -> None:
    """
    Sanity check wajib: pastikan ke-4 file kebaca semua (bukan cuma 3
    karena typo path / file kebetulan gak ke-mount).
    Cek jumlah unique 'source_file' di metadata == expected_files.
    Raise assertion error kalau tidak sesuai -- jangan diam-diam lanjut
    dengan data yang gak lengkap.
    """
    unique_files = {doc.metadata.get("source_file") for doc in documents}
    assert len(unique_files) == expected_files, (
        f"Expected {expected_files} unique PDF files, but found {len(unique_files)}. "
        f"Unique files: {unique_files}"
    )
