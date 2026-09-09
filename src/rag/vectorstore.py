import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_PERSIST_DIR = "/kaggle/working/chroma_db"


def build_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """
    Load embedding model open-source (bge-m3), auto-detect device.

    Tricky part: bge-m3 di-training dengan cosine similarity sebagai
    objective, jadi embedding-nya harus di-normalize (L2 norm = 1) supaya
    cosine similarity berperilaku benar. Kalau tidak dinormalize, hasil
    similarity antar chunk bisa bias ke chunk yang magnitude vektornya
    kebetulan lebih besar, bukan yang paling relevan secara makna.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[WARNING] CUDA tidak terdeteksi, embedding akan jalan di CPU "
            "(jauh lebih lambat untuk 4 dokumen UU + child chunking)."
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str = "legal_docs",
    persist_directory: str = DEFAULT_PERSIST_DIR,
) -> Chroma:
    """
    Buat Chroma collection kosong, siap diisi lewat ParentDocumentRetriever
    nanti (retriever yang handle splitting + add_documents, bukan di sini).

    Tricky part: default distance metric Chroma itu L2, BUKAN cosine.
    Karena embedding_model di atas sudah dinormalize untuk cosine similarity,
    kita harus paksa collection-nya pakai 'hnsw:space': 'cosine' juga --
    kalau tidak, ada mismatch antara asumsi training embedding (cosine)
    dan metric yang dipakai saat search (L2), hasil retrieval bisa
    sedikit off dari yang seharusnya.

    persist_directory default ke /kaggle/working supaya collection selamat
    kalau kernel restart di sesi yang sama (masuk tab Output Kaggle).
    JANGAN commit folder ini ke GitHub -- ini binary sqlite + index besar,
    dan bisa di-rebuild ulang dari 4 PDF kapan saja. Pastikan
    'chroma_db/' ditambahkan ke .gitignore.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
