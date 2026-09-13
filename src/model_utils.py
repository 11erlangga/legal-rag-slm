"""
model_utils.py

Utility untuk load model + tokenizer (Unsloth) dan setup PEFT/LoRA,
dipakai bareng di semua notebook eksperimen SFT.
"""

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_model_and_tokenizer(
    model_name: str,
    chat_template: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """
    Load base model (quantized) + tokenizer, lalu apply chat template.

    NOTE: nama chat_template harus PERSIS cocok dengan salah satu key di
    unsloth.chat_templates.CHAT_TEMPLATES -- cek dulu dengan:
        from unsloth.chat_templates import CHAT_TEMPLATES
        print(list(CHAT_TEMPLATES.keys()))
    sebelum ganti model_name ke family lain, karena template harus
    sesuai model (misal Qwen2.5 -> "qwen-2.5", Llama-3.1 -> "llama-3.1").
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        dtype=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules=None,
):
    """
    Apply LoRA adapter ke model. Parameter dibuat jadi argumen (bukan
    hardcoded) supaya gampang dibedakan antar eksperimen hyperparameter
    (mis. eksperimen 1 pakai r=16, eksperimen 2 coba r=32) tanpa duplikasi
    seluruh fungsi.
    """
    if target_modules is None:
        # default: cover attention (q,k,v,o) + FFN (gate,up,down)
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model


def run_coldstart_sft(
    model,
    tokenizer,
    coldstart_dataset,
    output_dir="coldstart_checkpoint",
    max_steps=150,
    seed=1010,
):
    """
    Mini-SFT singkat di atas checkpoint run1 (LoRA sudah attached).
    max_steps sengaja kecil — tujuan cuma inject prior format, bukan re-training penuh.
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        seed=seed,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=coldstart_dataset,
        args=sft_config,
    )

    trainer.train()
    return trainer
