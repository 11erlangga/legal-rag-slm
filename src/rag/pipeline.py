from IPython.display import Markdown, display
from langchain_core.documents import Document

from src.rag.ingestion import load_pdfs, validate_pdf_count
from src.rag.chunking import build_splitters, log_chunking_config
from src.rag.vectorstore import build_embedding_model, build_vectorstore
from src.rag.retrievers import (
    build_dense_retriever,
    ingest_into_dense_retriever,
    build_bm25_retriever,
    build_ensemble_retriever,
    build_reranked_retriever,
    print_retrieved_docs,
)
from src.rag.generation import (
    load_finetuned_model,
    build_text_generation_pipeline,
    format_context,
    build_prompt_runnable,
    SYSTEM_PROMPT_RAG,
)

VALID_RETRIEVER_MODES = ("dense", "hybrid", "hybrid_rerank")


class RAGPipeline:
    """
    Bungkus retriever + llm + prompt jadi satu objek dengan interface
    sederhana: generate(query) -> {"answer": str, "sources": list[Document]}.

    Sengaja BUKAN pakai LCEL chain (`|`) murni -- karena kita butuh akses
    eksplisit ke `docs` hasil retrieval (untuk ditampilkan sebagai sitasi
    terpisah dari jawaban), dan LCEL chain murni bikin retriever di-invoke
    2x (sekali di dalam chain, sekali lagi manual buat nampilin sumber) --
    boros, dan berisiko dapat hasil retrieval yang beda kalau ada
    non-determinism. Di sini retrieval cuma dipanggil sekali per query.
    """

    def __init__(self, retriever, llm, tokenizer, system_prompt: str = SYSTEM_PROMPT_RAG):
        self.retriever = retriever
        self.llm = llm
        self.prompt_runnable = build_prompt_runnable(tokenizer, system_prompt)

    def generate(self, query: str) -> dict:
        docs: list[Document] = self.retriever.invoke(query)
        context = format_context(docs)
        prompt = self.prompt_runnable.invoke({"context": context, "question": query})
        raw_output = self.llm.invoke(prompt)
        return {"answer": raw_output.strip(), "sources": docs}


def build_retrievers(
    pdf_dir: str,
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
    reranker_top_n: int = 3,
) -> dict:
    """
    Bangun ketiga retriever_mode SEKALIGUS dari SATU proses ingestion
    (load PDF, chunking, embedding, ingest ke dense retriever) -- bukan
    diulang per mode seperti desain sebelumnya.

    FIX untuk OOM: build_pipeline() versi sebelumnya dipanggil 3x terpisah
    di notebook untuk bandingin retriever_mode, dan tiap panggilan itu
    nge-RELOAD GENERATOR MODEL dari HF Hub -- padahal generator sama
    sekali gak dibutuhkan untuk ablation retrieval (sanity_check_retrieval
    cuma invoke retriever, gak pernah invoke llm). Akibatnya 3 instance
    generator model (+ embedding model) numpuk di GPU memory tanpa pernah
    dibebaskan, sampai OOM pas load instance ke-3.

    Solusi: pisahkan proses build retriever (murah, gak butuh generator)
    dari load generator (mahal, sekali aja). Bonus: "dense", "hybrid", dan
    "hybrid_rerank" sebenarnya bertingkat (hybrid dibangun DI ATAS
    dense_retriever yang sama, hybrid_rerank DI ATAS hybrid yang sama) --
    jadi PDF+embedding cukup diproses sekali, dipakai bersama ketiganya,
    bukan re-embed dokumen yang sama 3x.

    Return: dict {"dense": ..., "hybrid": ..., "hybrid_rerank": ...}
    """
    documents = load_pdfs(pdf_dir)
    validate_pdf_count(documents, expected_files=4)

    parent_splitter, child_splitter = build_splitters()
    log_chunking_config()

    embedding_model = build_embedding_model()
    vectorstore = build_vectorstore(embedding_model)

    dense_retriever = build_dense_retriever(vectorstore, parent_splitter, child_splitter)
    ingest_into_dense_retriever(dense_retriever, documents)

    bm25_retriever = build_bm25_retriever(documents, child_splitter)
    hybrid_retriever = build_ensemble_retriever(bm25_retriever, dense_retriever, ensemble_weights)

    hybrid_rerank_retriever = build_reranked_retriever(hybrid_retriever, top_n=reranker_top_n)

    return {
        "dense": dense_retriever,
        "hybrid": hybrid_retriever,
        "hybrid_rerank": hybrid_rerank_retriever,
    }


def build_generator(hf_repo_id: str, hf_token: str | None = None):
    """
    Load generator SEKALI, dipakai ulang untuk retriever_mode manapun.
    Jangan panggil berkali-kali dalam satu sesi kernel kecuali memang mau
    ganti model (misal eksperimen run1 vs run2) -- ini komponen paling
    berat di GPU memory.
    """
    model, tokenizer = load_finetuned_model(hf_repo_id, hf_token=hf_token)
    llm = build_text_generation_pipeline(model, tokenizer)
    return llm, tokenizer


def build_pipeline(
    pdf_dir: str,
    hf_repo_id: str,
    retriever_mode: str = "hybrid_rerank",
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
    reranker_top_n: int = 3,
    hf_token: str | None = None,
) -> RAGPipeline:
    """
    Convenience wrapper: bangun SATU retriever_mode + generator jadi
    RAGPipeline siap pakai. Cocok untuk pemakaian TUNGGAL (misal section
    "Full Pipeline untuk Interactive Use" di notebook).

    Untuk BANDINGIN beberapa retriever_mode sekaligus (ablation study),
    JANGAN panggil fungsi ini berkali-kali -- pakai build_retrievers() +
    build_generator() terpisah, supaya PDF gak di-ingest ulang dan
    generator gak di-load ulang tiap mode (itu penyebab OOM sebelumnya).
    """
    if retriever_mode not in VALID_RETRIEVER_MODES:
        raise ValueError(
            f"retriever_mode harus salah satu dari {VALID_RETRIEVER_MODES}, "
            f"dapat: {retriever_mode!r}"
        )

    retrievers = build_retrievers(pdf_dir, ensemble_weights, reranker_top_n)
    retriever = retrievers[retriever_mode]

    llm, tokenizer = build_generator(hf_repo_id, hf_token=hf_token)

    return RAGPipeline(retriever=retriever, llm=llm, tokenizer=tokenizer)


def sanity_check_retrieval(retriever, query: str) -> None:
    """
    Verifikasi retriever jalan -- requirement eksplisit Basic ("uji
    retrieval pada query relevan, tampilkan hasil chunk").

    FIX: terima retriever LANGSUNG (bukan RAGPipeline seperti sebelumnya)
    -- verifikasi retrieval gak butuh generator sama sekali, jadi bisa
    dipanggil murah tanpa perlu build_pipeline() penuh (yang otomatis
    ikut load generator).
    """
    docs = retriever.invoke(query)
    print(f"Query: {query}\n")
    print_retrieved_docs(docs)


def interactive_loop(pipeline: RAGPipeline) -> None:
    """
    Interface wajib Basic: input() + IPython.display.Markdown.

    Ketik 'exit' atau 'quit' untuk keluar dari loop -- tanpa exit
    condition, ini technically infinite loop yang cuma bisa dihentikan
    dengan interrupt kernel, kurang enak untuk demo ke penilai.
    """
    print("Legal AI Assistant -- ketik 'exit' atau 'quit' untuk keluar.\n")
    while True:
        query = input("Pertanyaan: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Selesai.")
            break
        if not query:
            continue

        result = pipeline.generate(query)

        display(Markdown(f"**Jawaban:**\n\n{result['answer']}"))

        sources_md = "\n".join(
            f"{i}. {doc.metadata.get('source_file', '?')}, "
            f"halaman {doc.metadata.get('page', '?')}"
            for i, doc in enumerate(result["sources"], start=1)
        )
        display(Markdown(f"**Sumber Referensi:**\n\n{sources_md}"))