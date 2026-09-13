"""
data_utils.py

Utility untuk load dan format dataset Alpaca-GPT4-Indonesian menjadi
format Chat Template (via tokenizer Unsloth), dipakai bareng di semua
notebook eksperimen SFT supaya logic mapping-nya konsisten & gak
copy-paste ulang tiap eksperimen.
"""

import random

from datasets import Dataset, load_dataset

DATASET_NAME = "Ichsan2895/alpaca-gpt4-indonesian"

THINK_TEMPLATES = [
    "Pertanyaan ini meminta saya untuk {task_hint}. Saya akan menjawab secara langsung dan jelas.",
    "Untuk menjawab ini, saya perlu memahami inti permintaan terkait {task_hint}. Berikut jawabannya.",
    "Saya akan menyusun jawaban berdasarkan konteks yang diberikan terkait {task_hint}.",
    "Permintaan ini berkaitan dengan {task_hint}. Saya akan memberikan jawaban yang relevan dan ringkas.",
    "Berdasarkan instruksi mengenai {task_hint}, saya akan menyusun respons yang sesuai.",
    "Saya perlu mempertimbangkan {task_hint} sebelum memberikan jawaban akhir.",
]


def load_split_dataset(test_size: float = 0.05, seed: int = 1010):
    """
    Load dataset mentah dari HuggingFace, lalu split train/val.

    NOTE: dataset ini cuma punya kolom ['Unnamed: 0', 'input', 'output'] --
    BUKAN format Alpaca standar 3-kolom (instruction/input/output). Kolom
    'input' di sini isinya instruksi/pertanyaan itu sendiri, bukan context
    tambahan. Kolom 'Unnamed: 0' adalah artifact index dari CSV export,
    di-drop karena gak dipakai.

    seed di-fix biar split konsisten & reproducible antar eksperimen
    (supaya perbandingan hyperparameter fair -- data train/val-nya sama).
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    split = dataset.train_test_split(test_size=test_size, seed=seed)

    train_dataset = split["train"].remove_columns(["Unnamed: 0"])
    val_dataset = split["test"].remove_columns(["Unnamed: 0"])

    return train_dataset, val_dataset


def build_formatting_func(tokenizer, system_prompt: str):
    """
    Return formatting function yang siap dipakai di dataset.map(batched=True).

    Dipisah jadi factory function (bukan formatting_func langsung) karena
    tokenizer & system_prompt beda-beda tergantung model/eksperimen yang
    lagi jalan -- jadi tiap notebook tinggal panggil dengan tokenizer &
    system_prompt masing-masing.
    """

    def formatting_prompts_func(examples):
        user_inputs = examples["input"]
        outputs = examples["output"]
        texts = []

        for user_input, output in zip(user_inputs, outputs):
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input.strip()},
                {"role": "assistant", "content": output.strip()},
            ]
            text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        return {"text": texts}

    return formatting_prompts_func


def prepare_datasets(
    tokenizer, system_prompt: str, test_size: float = 0.05, seed: int = 1010
):
    """
    One-call helper: load, split, dan format dataset sekaligus.

    Return: (train_dataset, val_dataset) -- keduanya sudah punya kolom "text"
    siap dipakai SFTTrainer.
    """
    train_dataset, val_dataset = load_split_dataset(test_size=test_size, seed=seed)
    formatting_func = build_formatting_func(tokenizer, system_prompt)

    train_dataset = train_dataset.map(formatting_func, batched=True)
    val_dataset = val_dataset.map(formatting_func, batched=True)

    return train_dataset, val_dataset


def extract_task_hint(instruction: str, max_words: int = 6) -> str:
    """
    Ambil ringkasan singkat dari instruksi untuk dipakai di <think> placeholder.
    Heuristik: buang kata perintah umum di depan, ambil sisa kalimat pendek.
    """
    instruction = instruction.strip()
    # Ambil klausa pertama sebelum newline
    first_line = instruction.split("\n")[0]
    words = first_line.split()
    hint = " ".join(words[:max_words]).rstrip(".,:;")
    return hint.lower() if hint else "permintaan pengguna"


def build_coldstart_example(row, tokenizer, system_prompt: str):
    task_hint = extract_task_hint(row["input"])
    think_content = random.choice(THINK_TEMPLATES).format(task_hint=task_hint)

    assistant_content = f"<think>\n{think_content}\n</think>\n{row['output']}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def build_coldstart_dataset(
    base_dataset, tokenizer, system_prompt: str, n_samples: int = 300, seed: int = 1010
) -> Dataset:
    subset = base_dataset.shuffle(seed=seed).select(
        range(min(n_samples, len(base_dataset)))
    )
    coldstart = subset.map(
        lambda row: build_coldstart_example(row, tokenizer, system_prompt),
        remove_columns=subset.column_names,
    )
    return coldstart
