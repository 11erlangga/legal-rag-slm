import torch
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

SYSTEM_PROMPT_RAG = (
    "Kamu adalah asisten AI Legal Team yang menjawab pertanyaan hukum "
    "HANYA berdasarkan konteks dokumen yang diberikan. "
    "Jika jawaban tidak ditemukan dalam konteks, katakan dengan jujur bahwa "
    "kamu tidak menemukan informasinya di dokumen -- jangan berspekulasi atau "
    "menambahkan pengetahuan di luar konteks. "
    "Jawab dalam Bahasa Indonesia, singkat dan jelas, dan sebutkan sumber "
    "(nama dokumen/halaman) yang mendasari jawabanmu kalau relevan."
)


def load_finetuned_model(
    hf_repo_id: str,
    load_in_4bit: bool = True,
    hf_token: str | None = None,
):
    """
    Load model hasil fine-tuning SENDIRI dari HF Hub (bukan base model dari
    penyedia lain) -- ini fix untuk blocker Kriteria Reject sebelumnya.

    hf_repo_id: HARUS repo hasil push_to_hub_merged kamu sendiri
    (contoh: "username/sft-qwen25-3b-run2"), cek link_huggingface.txt.

    Quantization 4-bit dipakai untuk inference juga (bukan cuma training)
    supaya muat nyaman di T4 bareng embedding model + reranker yang jalan
    di sesi yang sama -- kalau GPU memory ketat, ini yang paling murah
    untuk dilonggarkan duluan.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_repo_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo_id,
        quantization_config=bnb_config if load_in_4bit else None,
        device_map="auto",
        token=hf_token,
    )

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer dari {hf_repo_id} tidak punya chat_template. "
            f"Ini harusnya ke-carry otomatis dari push_to_hub_merged saat "
            f"training (chat template 'qwen-2.5'). Cek ulang proses push."
        )

    return model, tokenizer


def build_text_generation_pipeline(
    model,
    tokenizer,
    max_new_tokens: int = 1000,
    temperature: float = 0.2,
) -> HuggingFacePipeline:
    """
    do_sample=True + temperature rendah (0.2): sedikit variasi tapi tetap
    conservative -- cocok untuk legal QA yang butuh presisi.

    Alternatif do_sample=False (greedy) lebih deterministic/reproducible,
    yang buat legal domain sebenarnya defensible juga (jawaban gak boleh
    "beda-beda" tiap ditanya ulang dengan pertanyaan sama). Saya pilih
    tetap do_sample=True dulu konsisten sama pattern awal kamu -- kalau
    nanti testing GRPO/RAG nunjukin jawaban suka goyang antar run untuk
    query yang sama, ini parameter pertama yang saya sarankan diubah ke
    do_sample=False.
    """
    text_gen_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def format_context(docs: list[Document]) -> str:
    """
    Gabungkan Document hasil retrieval jadi satu string bersih untuk
    disisipkan ke prompt -- dengan label sumber per chunk.

    FIX untuk bug di chain lama: sebelumnya {context} langsung diisi
    list of Document mentah dari retriever (retriever dipasang langsung
    sebagai value dict, bukan di-pipe lewat formatter) -- itu bikin prompt
    berisi representasi Python object (termasuk noise metadata) alih-alih
    teks bersih. Fungsi ini yang seharusnya berdiri di antara retriever dan
    prompt template di dalam chain.

    Label sumber per chunk juga jadi fondasi untuk citation requirement
    (Skilled) nanti -- model bisa "lihat" dari dokumen/halaman mana tiap
    potongan konteks berasal, bukan cuma teks polos tanpa atribusi.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "?")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Sumber: {source}, halaman {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_prompt_runnable(
    tokenizer, system_prompt: str = SYSTEM_PROMPT_RAG
) -> RunnableLambda:
    """
    Return Runnable yang terima dict {"context": str, "question": str}
    dan hasilkan prompt string via tokenizer.apply_chat_template().

    FIX untuk bug prompt template hardcode token Llama-3 sebelumnya:
    dengan apply_chat_template, format token (ChatML untuk Qwen2.5, apapun
    format Llama kalau kamu ganti model dasar) otomatis mengikuti tokenizer
    model kamu sendiri -- gak ada lagi hardcoded string token yang bisa
    mismatch sama model yang sebenarnya dipakai.
    """

    def _format(inputs: dict) -> str:
        context = inputs["context"]
        question = inputs["question"]
        user_content = f"Konteks:\n{context}\n\nPertanyaan: {question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return RunnableLambda(_format)
