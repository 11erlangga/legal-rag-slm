from IPython.display import Markdown, display
from langchain_core.documents import Document

from src.rag.chunking import build_splitters, log_chunking_config
from src.rag.generation import (
    SYSTEM_PROMPT_RAG,
    build_prompt_runnable,
    build_text_generation_pipeline,
    format_context,
    load_finetuned_model,
)
from src.rag.ingestion import load_pdfs, validate_pdf_count
from src.rag.retrievers import (
    build_bm25_retriever,
    build_dense_retriever,
    build_ensemble_retriever,
    build_reranked_retriever,
    ingest_into_dense_retriever,
    print_retrieved_docs,
)
from src.rag.vectorstore import build_embedding_model, build_vectorstore

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

    def __init__(
        self, retriever, llm, tokenizer, system_prompt: str = SYSTEM_PROMPT_RAG
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_runnable = build_prompt_runnable(tokenizer, system_prompt)

    def generate(self, query: str) -> dict:
        docs: list[Document] = self.retriever.invoke(query)
        context = format_context(docs)
        prompt = self.prompt_runnable.invoke({"context": context, "question": query})
        raw_output = self.llm.invoke(prompt)
        return {"answer": raw_output.strip(), "sources": docs}


def build_pipeline(
    pdf_dir: str,
    hf_repo_id: str,
    retriever_mode: str = "hybrid_rerank",
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
    reranker_top_n: int = 3,
    hf_token: str | None = None,
) -> RAGPipeline:
    """
    Wiring penuh: PDF -> chunking -> embedding -> vectorstore -> retriever
    (sesuai retriever_mode) -> model fine-tuning -> RAGPipeline siap pakai.

    retriever_mode:
      - "dense": cuma ParentDocumentRetriever (semantic only) -- setara
        requirement Basic ("uji retrieval pada query relevan").
      - "hybrid": + BM25 via EnsembleRetriever -- setara Skilled.
      - "hybrid_rerank": + CrossEncoderReranker -- setara Advanced
        (minus HyDE & fallback DuckDuckGo, itu ditambahkan terpisah nanti
        di atas retriever "hybrid_rerank" ini, bukan gantiin).

    Kenapa satu fungsi bisa hasilin ketiga level itu (bukan tiga fungsi
    terpisah): supaya bisa jalanin ketiganya dengan PDF & embedding yang
    SAMA persis dalam satu sesi notebook, buat ablation study -- "apakah
    hybrid beneran lebih baik dari dense-only, apakah reranker beneran
    worth latency-nya" -- itu jauh lebih meyakinkan sebagai bukti kalau
    dibandingkan pakai data identik, bukan run terpisah-pisah.
    """
    if retriever_mode not in VALID_RETRIEVER_MODES:
        raise ValueError(
            f"retriever_mode harus salah satu dari {VALID_RETRIEVER_MODES}, "
            f"dapat: {retriever_mode!r}"
        )

    # 1. Ingestion
    documents = load_pdfs(pdf_dir)
    validate_pdf_count(documents, expected_files=4)

    # 2. Chunking
    parent_splitter, child_splitter = build_splitters()
    log_chunking_config(parent_splitter, child_splitter)

    # 3. Embedding + vectorstore
    embedding_model = build_embedding_model()
    vectorstore = build_vectorstore(embedding_model)

    # 4. Dense retriever (selalu dibangun, jadi basis untuk mode lain juga)
    dense_retriever = build_dense_retriever(
        vectorstore, parent_splitter, child_splitter
    )
    ingest_into_dense_retriever(dense_retriever, documents)

    retriever = dense_retriever

    if retriever_mode in ("hybrid", "hybrid_rerank"):
        bm25_retriever = build_bm25_retriever(documents, child_splitter)
        retriever = build_ensemble_retriever(
            bm25_retriever, dense_retriever, ensemble_weights
        )

    if retriever_mode == "hybrid_rerank":
        retriever = build_reranked_retriever(retriever, top_n=reranker_top_n)

    # 5. Generation model (hasil fine-tuning sendiri)
    model, tokenizer = load_finetuned_model(hf_repo_id, hf_token=hf_token)
    llm = build_text_generation_pipeline(model, tokenizer)

    return RAGPipeline(retriever=retriever, llm=llm, tokenizer=tokenizer)


def sanity_check_retrieval(pipeline: RAGPipeline, query: str) -> None:
    """
    Panggil manual setelah build_pipeline() untuk verifikasi retriever
    jalan sebelum masuk interactive loop -- requirement eksplisit Basic
    ("uji retrieval pada query relevan, tampilkan hasil chunk").
    """
    docs = pipeline.retriever.invoke(query)
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
