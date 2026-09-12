## Catatan Keputusan: Penentuan `max_steps` SFT

### Masalah
Brief mewajibkan `SFTTrainer` jalan minimal 800 steps. Tapi 800 itu batas minimum,
bukan target — perlu dihitung angka yang proporsional dengan waktu training yang
tersedia dan kuota GPU Kaggle (~30 jam/minggu, sesi maksimal 9-12 jam).

### Metode
1. Jalankan training percobaan (`max_steps=20`, `eval_strategy="no"`,
   `save_strategy="no"` — dimatikan sementara supaya tidak mengganggu pengukuran
   kecepatan murni) untuk mengukur kecepatan training di GPU Kaggle (T4/P100).
2. Progress bar HuggingFace Trainer menunjukkan kecepatan **~0.26 it/s**
   (rata-rata dari 20 steps), dikonversi jadi **~3.85 detik/step**.
3. Dari situ, estimasi jumlah steps yang muat dalam beberapa skenario durasi:

   | Durasi | Steps |
   |--------|-------|
   | 1 jam | ~935 |
   | 1.5 jam | ~1.558 |
   | 2.5 jam | ~2.337 |

   (dataset training ±45.000 baris, effective batch size 8 → 1 epoch penuh ≈ 5.600
   steps ≈ ~6 jam)

### Keputusan
Dipilih **`max_steps=2000`** (~2.1 jam per run), dengan pertimbangan:
- Jauh di atas minimum brief (800 steps), cukup untuk menghasilkan loss curve yang
  informatif saat membandingkan 2 eksperimen hyperparameter (syarat Skilled).
- Proses ini perlu dijalankan **2x** (2 eksperimen berbeda), jadi total waktu SFT
  ≈ 4.2 jam — masih menyisakan kuota mingguan yang cukup besar untuk tahap GRPO,
  yang per step jauh lebih berat karena melibatkan multiple generation
  (`num_generations`) per step.
- Tidak mengambil full 1 epoch (~5.600 steps) karena akan memakan kuota berlebihan
  jika dikalikan 2 eksperimen, tanpa jaminan peningkatan kualitas yang sepadan pada
  tahap SFT ini (perbaikan kualitas lebih banyak diharapkan datang dari tahap GRPO
  dan RAG context injection, bukan durasi SFT semata).

### Turunan
- `eval_steps=100`, `save_steps=100` — ditentukan sebagai ~5% dari `max_steps`,
  memberi ±20 titik checkpoint/evaluasi sepanjang training: cukup sering untuk
  memantau tren loss dan sebagai mitigasi jika sesi Kaggle terputus, tanpa
  membebani waktu training dengan evaluasi yang terlalu sering.


  Oke, opsi A murni (threshold tanpa repetition-check). Ini draft yang bisa langsung kamu taruh sebagai markdown cell di Section 2.7 (`ROUGE_SIMILARITY_THRESHOLD_FINAL`). Saya tulis dengan nada yang jujur soal trade-off-nya — jangan dihapus bagian keterbatasannya, itu justru yang bikin evaluasi ini kredibel.

---

### 2.7 — Penentuan `ROUGE_SIMILARITY_THRESHOLD_FINAL`

**Keputusan: `ROUGE_SIMILARITY_THRESHOLD_FINAL = 0.2`**

**Metodologi:**
Dari 20 sample eval set (greedy decoding), dilakukan manual review terhadap seluruh sample dengan ROUGE-L fmeasure < 0.6 (19 dari 20 sample) untuk memastikan skor rendah benar-benar mencerminkan jawaban yang salah, bukan sekadar paraphrase valid atau keterbatasan metrik ROUGE. Setiap sample dikategorikan ke salah satu dari 4 kelas:

1. **Data rusak/unanswerable** — instruksi merujuk data yang tidak diberikan, atau artefak error translation pipeline ikut ter-inject ke teks instruksi. Reward rendah tidak terhindarkan di threshold berapa pun.
2. **Open-ended / instruksi ambigu** — ground truth hanya salah satu dari banyak jawaban valid, atau instruksi itu sendiri ambigu sehingga model dan ground truth menginterpretasikannya secara berbeda tapi sama-sama masuk akal.
3. **Paraphrase substansi benar** — isi jawaban model secara faktual/konseptual sama dengan ground truth, hanya beda kata/struktur kalimat.
4. **Model benar-benar salah** — ditemukan pola kegagalan konkret: repetition/degenerate output (model terjebak mengulang kalimat yang sama berkali-kali), pelanggaran constraint eksplisit di instruksi (mis. diminta rentang harga tertentu, jawaban di luar rentang itu), atau kesalahan faktual terhadap isi yang ditanyakan.

**Temuan kunci: kategori 3 dan 4 tumpang-tindih di rentang skor yang sama (~0.16–0.28).**
Contoh konkret:
- Kategori 3 (paraphrase valid) ditemukan di skor serendah 0.186 (penjelasan arsitektur CNN dengan istilah berbeda tapi konsep identik).
- Kategori 4 (model salah) ditemukan di skor setinggi 0.277 (kesalahan faktual pada isi cerita Romeo & Juliet), termasuk kasus repetition collapse di skor 0.161 dan pelanggaran constraint eksplisit di skor 0.231.

Artinya **ROUGE-L pada rentang ini tidak bisa membedakan secara bersih** antara "jawaban benar dengan kata-kata berbeda" dan "jawaban yang gagal". Ini adalah keterbatasan inheren dari metrik lexical-overlap seperti ROUGE, bukan bug pada implementasi reward function.

**Alasan threshold dipilih di 0.2 (bukan lebih tinggi):**
Prioritas diberikan untuk **tidak mengorbankan sinyal reward positif** bagi jawaban yang benar (kategori 3), karena `correctness_reward_func` adalah satu-satunya dari 4 reward function yang mengukur kebenaran isi jawaban — kalau threshold dinaikkan untuk mengejar "keamanan" dari kategori 4, risikonya reward function ini jadi nyaris selalu bernilai 0 untuk mayoritas sample yang sebenarnya benar, sehingga GRPO kehilangan sinyal pembeda yang justru paling penting.

**Trade-off yang diterima secara sadar (known limitation):**
Dengan threshold 0.2, minoritas sample kategori 4 yang skornya kebetulan berada tepat di bawah 0.2 (dalam sample review ini: 1 dari 20 sample, kasus repetition collapse) berpotensi tetap mendapat reward correctness yang salah (dianggap benar padahal gagal). Ini diterima sebagai noise yang proporsional untuk level project ini, mengingat:
- Reward correctness hanyalah 1 dari 4 sinyal yang membentuk total reward GRPO (bukan satu-satunya sinyal kualitas).
- Menutup gap ini sepenuhnya membutuhkan mekanisme deteksi tambahan (mis. deteksi repetisi n-gram, constraint-checking per instruksi) yang berada di luar scope reward function correctness sesuai brief, dan menambah kompleksitas yang belum tentu sepadan dengan ukuran masalah (~5% dari sample review).

**Catatan tambahan:** temuan ini juga konsisten dengan catatan kualitas dataset sebelumnya (`alpaca-gpt4-indonesian` mengandung instruksi rusak akibat artefak translation) — noise floor pada `correctness_reward_func` sudah diketahui bersifat tidak nol secara struktural, dan hasil di atas mengonfirmasi bahwa noise tersebut berasal dari lebih dari satu sumber (kualitas data + keterbatasan metrik ROUGE itu sendiri).