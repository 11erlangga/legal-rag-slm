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