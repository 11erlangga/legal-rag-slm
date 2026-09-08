"""
data_utils.py

Utility untuk load dan format dataset Alpaca-GPT4-Indonesian menjadi
format Chat Template (via tokenizer Unsloth), dipakai bareng di semua
notebook eksperimen SFT supaya logic mapping-nya konsisten & gak
copy-paste ulang tiap eksperimen.
"""

from datasets import load_dataset

DATASET_NAME = "Ichsan2895/alpaca-gpt4-indonesian"


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
