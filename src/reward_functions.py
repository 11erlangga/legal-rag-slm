"""
reward_functions.py

Kumpulan reward function custom untuk GRPOTrainer (TRL + Unsloth) pada
Kriteria 1 - Advanced (Legal RAG SLM project).

4 reward function sesuai brief:
1. format_reward_func            -> reward shaping tag <think>...</think>, max +1.0
2. reasoning_length_reward_func  -> reward proporsional panjang isi <think>
3. correctness_reward_func       -> reward berdasarkan ground truth (containment / ROUGE-L)
4. language_reward_func          -> reward berdasarkan bahasa output akhir (ID vs EN)

Semua function mengikuti signature yang diharapkan GRPOTrainer:
    def reward_func(completions, **kwargs) -> list[float]

Catatan penting (lihat PROGRESS_LOG.md untuk detail & histori keputusan):
- ROUGE_SIMILARITY_THRESHOLD di bawah masih PLACEHOLDER, belum di-tuning
  berdasarkan distribusi data asli. Akan diselesaikan bersamaan dengan
  perbandingan loss curve eksperimen 1 vs 2 (lihat Next Steps di PROGRESS_LOG.md).
- language_reward_func hanya menilai bahasa pada FINAL ANSWER (setelah
  </think>), bukan seluruh teks termasuk isi reasoning. Ini keputusan
  eksplisit karena brief hanya menyebut "output akhir" -- lihat catatan
  di PROGRESS_LOG.md soal implikasi/risiko keputusan ini.
"""

from langdetect import LangDetectException, detect
from rouge_score import rouge_scorer

# =============================================================================
# Konstanta
# =============================================================================

ROUGE_SIMILARITY_THRESHOLD = 0.2

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


# =============================================================================
# Helper (dipakai bersama oleh beberapa reward function -- definisikan sekali)
# =============================================================================


def extract_final_answer(text: str) -> str:
    """
    Ambil jawaban akhir = semua teks setelah </think> yang PERTAMA.
    Kalau tidak ada </think> sama sekali, anggap seluruh teks adalah jawaban
    (model belum/tidak memakai format reasoning, tapi tetap dinilai).
    """
    close_idx = text.find("</think>")
    if close_idx == -1:
        return text.strip()
    return text[close_idx + len("</think>") :].strip()


def _get_completion_text(completion) -> str:
    """
    Ekstrak string content dari satu elemen `completions` GRPOTrainer.
    Format completions biasanya: [[{"role": ..., "content": "..."}], ...]
    """
    return completion[0]["content"]


# =============================================================================
# 1. format_reward_func
# =============================================================================


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward shaping untuk format <think>...</think> sebelum jawaban akhir.

    Skema (max +1.0):
    - Perfect format (mulai <think>, ditutup benar, diikuti jawaban akhir,
      masing-masing tag muncul tepat 1x) -> +1.0 (EXCLUSIVE, tidak ditumpuk
      dengan reward parsial di bawah)
    - Kalau tidak perfect, reward parsial (additive):
        - startswith <think> -> +0.2
        - ada </think> di suatu tempat -> +0.3
    - Penalty -0.5 kalau <think> atau </think> muncul >1 kali (halusinasi),
      diterapkan independen dari skema di atas.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        reward = 0.0

        open_count = text.count("<think>")
        close_count = text.count("</think>")
        starts_with_think = text.strip().startswith("<think>")
        has_closing = close_count >= 1

        followed_by_answer = False
        if has_closing:
            after_close = text.split("</think>", 1)[1].strip()
            followed_by_answer = len(after_close) > 0

        is_perfect = (
            starts_with_think
            and has_closing
            and followed_by_answer
            and open_count == 1
            and close_count == 1
        )

        if is_perfect:
            reward = 1.0
        else:
            if starts_with_think:
                reward += 0.2
            if has_closing:
                reward += 0.3

        if open_count > 1 or close_count > 1:
            reward -= 0.5

        rewards.append(reward)

    return rewards


# =============================================================================
# 2. reasoning_length_reward_func
# =============================================================================


def reasoning_length_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward proporsional terhadap panjang isi <think>...</think>.
    Toleran kalau reasoning terpotong oleh token limit (artinya <think> ada
    tapi </think> belum sempat muncul karena max_completion_length kepotong).

    Skema:
    - Tidak ada tag / isi kosong -> 0.0
    - <50 karakter                -> 0.2
    - 50-199 karakter              -> 0.5
    - >=200 karakter               -> 1.0

    Catatan: hanya menghitung pasangan <think>...</think> PERTAMA. Kasus
    <think> muncul berkali-kali (halusinasi) sudah dihukum terpisah di
    format_reward_func, sehingga di sini tidak perlu di-double-handle.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)

        open_idx = text.find("<think>")
        if open_idx == -1:
            rewards.append(0.0)
            continue

        content_start = open_idx + len("<think>")
        close_idx = text.find("</think>", content_start)

        if close_idx == -1:
            # <think> ada tapi </think> belum muncul -> kemungkinan terpotong
            # token limit. Toleran: anggap semua sisa text setelah <think>
            # sebagai isi reasoning yang belum sempat ditutup.
            reasoning_content = text[content_start:]
        else:
            reasoning_content = text[content_start:close_idx]

        length = len(reasoning_content.strip())

        if length == 0:
            reward = 0.0
        elif length < 50:
            reward = 0.2
        elif length < 200:
            reward = 0.5
        else:
            reward = 1.0

        rewards.append(reward)

    return rewards


# =============================================================================
# 3. correctness_reward_func
# =============================================================================


def correctness_reward_func(prompts, completions, output, **kwargs) -> list[float]:
    """
    +1.0 kalau jawaban akhir:
      (a) MENGANDUNG ground truth sebagai substring, ATAU
      (b) mirip secara ROUGE-L (fmeasure >= ROUGE_SIMILARITY_THRESHOLD)
    Dua kondisi independen (OR), sesuai dokumen asli.

    `answer` diharapkan berupa list ground truth (kolom Output dataset),
    urutannya sejajar dengan `completions`.
    """
    responses = [_get_completion_text(c) for c in completions]
    extracted_answers = [extract_final_answer(r) for r in responses]

    rewards = []
    for pred, gt in zip(extracted_answers, output):
        pred_norm = pred.strip().lower()
        gt_norm = gt.strip().lower()

        if len(pred_norm) == 0 or len(gt_norm) == 0:
            rewards.append(0.0)
            continue

        # Kondisi (a): containment
        contains_gt = gt_norm in pred_norm

        # Kondisi (b): similarity
        score = _rouge_scorer.score(gt_norm, pred_norm)
        rouge_l_f1 = score["rougeL"].fmeasure
        is_similar = rouge_l_f1 >= ROUGE_SIMILARITY_THRESHOLD

        rewards.append(1.0 if (contains_gt or is_similar) else 0.0)

    return rewards


# =============================================================================
# 4. language_reward_func
# =============================================================================


def language_reward_func(completions, **kwargs) -> list[float]:
    """
    Cek bahasa jawaban akhir (setelah </think>):
    - Murni Bahasa Indonesia -> +1.0
    - Tiba-tiba menjawab Bahasa Inggris -> -0.5
    - Bahasa lain / gagal dideteksi / jawaban kosong -> 0.0 (netral, brief
      hanya menyebut 2 skenario eksplisit ID dan EN)

    Catatan: hanya menilai bahasa pada FINAL ANSWER (setelah </think>),
    bukan seluruh teks termasuk isi reasoning. Lihat PROGRESS_LOG.md untuk
    diskusi risiko keputusan ini.
    """
    responses = [_get_completion_text(c) for c in completions]
    final_answers = [extract_final_answer(r) for r in responses]

    rewards = []
    for answer_text in final_answers:
        text = answer_text.strip()

        if len(text) == 0:
            rewards.append(0.0)
            continue

        try:
            detected_lang = detect(text)
        except LangDetectException:
            rewards.append(0.0)
            continue

        if detected_lang == "en":
            rewards.append(-0.5)
        elif detected_lang == "id":
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
