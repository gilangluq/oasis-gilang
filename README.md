# OASIS — OJK Agentic Self-Service Intelligence System (POC/MVP)

Aplikasi Streamlit untuk **bertanya dalam bahasa Indonesia** atas data agregat sektor jasa keuangan (8 subsektor), dengan
**katalog dari Kamus_Data**, **laporan kualitas data**, **analitik deterministik**, **perbandingan lintas sektor yang
sadar-satuan**, **deteksi “Perlu Perhatian”**, dan **proyeksi sederhana yang dapat dijelaskan**.

Berjalan penuh **tanpa API key/LLM**. LLM (Claude) hanya opsi tambahan untuk menafsirkan pertanyaan — seluruh angka tetap
dihitung kode atas data tervalidasi.

---

## 1. Menjalankan secara lokal (± 3 menit)

```bash
# Python 3.10–3.13
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # + requirements-dev.txt untuk pytest/LLM opsional

python scripts/generate_demo_workbook.py   # membuat data/demo/oasis_demo_simulasi.xlsx (sudah disertakan)
streamlit run app.py                        # buka http://localhost:8501
```

**Sumber data utama**: `data/public/Dataset_dengan_Metadata_dan_Kamus_Data.xlsx` — data statistik **publik**, disertakan
dalam repositori dan dipakai otomatis sebagai sumber default (“Data statistik publik” di sidebar, status data *Publik*),
termasuk pada deployment online. Alternatif: “Demo simulasi” atau “Unggah workbook”. Data internal (bila ada) letakkan di
`data/private/` (diabaikan git) atau tunjuk dengan `$env:OASIS_LOCAL_WORKBOOK="C:\path\file.xlsx"`; opsi ini tidak
muncul pada mode demo publik. Di Windows jalankan dengan `python -m streamlit run app.py`.

Nyatakan **status data** (Simulasi / Asli-internal / Publik / Belum dinyatakan) di sidebar — status ini tercatat pada asal
data setiap hasil.

Uji otomatis: `python -m pytest -q tests` (opsional uji terhadap workbook asli:
`OASIS_TEST_WORKBOOK=<path> python -m pytest -q tests`).

---

## 2. Skenario demo untuk penguji (± 10 menit)

| # | Klik contoh / ketik | Yang ditunjukkan |
|---|---|---|
| 1 | *Berapa NPL tertinggi?* | **Ambiguitas**: NPL ada di Bank Umum & BPR → OASIS menampilkan pilihan, tidak memilih diam-diam. |
| 2 | *Bandingkan tren Rasio NPL Bank Umum dan BPR 2023-2025* | Matriks: **Absolut** (rasio %, konsep & satuan sama) → satu sumbu; perubahan dalam **pp**; threshold simulasi putus-putus; anomali titik merah (klik). |
| 3 | *Total kredit dan NPL bank umum 24 bulan terakhir* | Satuan berbeda (Rp M vs %) → **dua sumbu** berlabel + catatan bahwa tinggi garis tidak sebanding. |
| 4 | *Bandingkan total aset asuransi jiwa dan asuransi umum* | Satuan **belum terkonfirmasi** di kamus → hanya **indeks basis 100**. Konfirmasi satuan di sidebar → matriks berubah menjadi Absolut (tercatat di log audit). |
| 5 | *Bagaimana hubungan IHSG dan total investasi dana pensiun?* | Scatter + korelasi **level vs perubahan**, label **eksploratif, bukan kausal**. |
| 6 | *Jumlah kredit bank umum tahun 2024* | Penolakan menjumlahkan **stok** antarbulan + alternatif yang valid. Bandingkan: *jumlah volume transaksi 2024* (arus → boleh). |
| 7 | *Proyeksi total aset BPR 12 bulan ke depan* | Backtest rolling-origin memilih metode; interval ~80%; peringatan lompatan struktural; label *bukan prediksi resmi OJK*. |
| 8 | *Apa yang perlu perhatian di perusahaan pembiayaan?* | Mode pindai seluruh indikator subsektor → kotak **Perlu Perhatian** berprioritas. |
| 9 | Tab **Kualitas Data** | Temuan nyata workbook (lihat §6), uji aturan eksplisit. |
| 10 | Tab **Lintas Sektor → Network** | Graph hanya dari aturan eksplisit; warna tepi = hasil uji data (mis. FAR Pembiayaan **gagal**). |

Setiap grafik memiliki tab **Data (grid)** berisi data persis yang diplot (unduh CSV). Grafik, tabel, kotak perhatian, dan
executive summary memakai **satu objek hasil** (`AnalysisResult`) dengan jendela periode yang sama.

---

## 3. Arsitektur

```
app.py                     UI Streamlit (tab: Tanya, Katalog, Kualitas, Kompatibilitas, Lintas Sektor, Proyeksi, Tata Kelola)
oasis/
  catalog.py               Baca Kamus_Data & Metadata_Sheet → katalog; klasifikasi satuan & jenis ukuran (stok/arus/rasio/indeks)
  data_prep.py             Baca sheet, parse Periode YYYYMMDD→akhir bulan, rekonsiliasi field kamus↔header, laporan kualitas, provenance
  nlu.py                   Parser pertanyaan berbasis aturan: periode, horizon, konsep/sinonim, subsektor, intent, ambiguitas
  llm.py                   (opsional) tafsir dengan Claude; keluaran divalidasi terhadap katalog; tanpa nilai data dikirim
  analytics.py             Statistik deskriptif, MoM/YoY (pp vs %), tren OLS, aturan penjumlahan, penyelarasan periode, korelasi
  compat.py                Matriks kompatibilitas: Absolut / Perubahan pp / Indeks 100 / Berdampingan
  attention.py             Anomali (skor-z robust MAD), threshold berlabel, prioritas & saran pemeriksaan
  forecast.py              Naive / seasonal naive / drift / tren; backtest rolling-origin; interval empiris; batas kewajaran
  engine.py                Orkestrasi pertanyaan → AnalysisResult (satu konteks data untuk semua tampilan)
  viz.py                   Plotly: garis, indeks 100, YoY, dua sumbu, berdampingan, batang, scatter, heatmap, proyeksi, network
  summary.py               Executive summary deterministik (templat)
config/
  concepts.json            Sinonim indikator & subsektor (tanpa menambah field)
  thresholds.json          Threshold SIMULASI (bukan batas regulasi)
  relations.json           Aturan hubungan eksplisit (dasar network & uji konsistensi)
  unit_conversions.json    Konversi satuan eksplisit & dapat diaudit (default kosong)
scripts/generate_demo_workbook.py   Pembangkit workbook simulasi berstruktur sama
tests/                     pytest (37 uji: parsing, NLU, kompatibilitas, analitik, proyeksi)
```

Alur: **Workbook → catalog + data_prep (validasi) → nlu (tafsir) → engine (analytics, compat, attention, forecast) → viz +
summary → UI**.

---

## 4. Aturan tata kelola yang ditegakkan kode

1. **Kamus = sumber kebenaran.** 89 indikator berasal dari 97 baris Kamus_Data (8 baris `Periode` = kunci waktu). Kolom sheet
   yang tidak ada di kamus diabaikan dan dilaporkan.
2. **Pemetaan field auditable.** Nama di kamus sering berbeda dengan header sheet (mis. kamus `Laba Asuransi Jiwa` vs header
   `Laba (Rp M)`). Metode: *cocok persis → cocok setelah normalisasi → posisi kolom (perlu konfirmasi)*, terlihat di Katalog.
3. **Satuan tidak diasumsikan.** Unit kamus `Tidak tercantum…` → **“satuan belum terkonfirmasi”** walau header sheet
   menulis `(Rp M)` (konflik dilaporkan). `Rp M` dan `Rp triliun` tidak dikonversi tanpa entri di `unit_conversions.json`.
   RBC tidak diperlakukan sebagai % karena satuannya tidak tercantum.
4. **Rasio %** → perubahan dalam **pp**; besaran positif → **%**; jenis belum pasti → selisih dalam satuan sumber.
5. **Stok tidak dijumlahkan antarbulan**; rasio/indeks juga tidak. Arus “selama periode” boleh. Seri terindikasi
   **kumulatif tahun berjalan** (reset Januari) ditandai; MoM-nya diberi peringatan dan anomalinya dihitung atas nilai
   de-kumulasi.
6. **MoM/YoY memakai bulan kalender yang tepat**; bulan hilang tidak “dilompati”.
7. **Threshold** default berlabel *threshold simulasi*; ubahan pengguna berlabel *konfigurasi pengguna*.
8. **Korelasi = eksploratif**; network hanya dari `relations.json` (+ tepi korelasi opsional bergaya berbeda).
9. **Proyeksi**: horizon 3/6/12; minimal `max(12, 3×horizon)` observasi kontigu; seri konstan/kosong ditolak; bukan
   prediksi resmi OJK.
10. **Asal data** di setiap hasil: sheet, field kamus, kolom sheet, metode pemetaan, periode terpakai, jumlah observasi,
    file + sha256, waktu unggah/penyusunan, status data. Log audit sesi dapat diunduh (tab Tata Kelola).

---

## 5. Deployment online (hanya data simulasi/publik)

> Jangan menyertakan data internal OJK, rahasia, atau API key ke repositori/demo publik. `.gitignore` mengecualikan semua `*.xlsx`
> kecuali data statistik publik (`data/public/`) dan workbook demo simulasi, serta `data/private/` dan
> `.streamlit/secrets.toml`.

**Streamlit Community Cloud (gratis, paling cepat)**
1. Push repo ini ke GitHub (publik/privat).
2. share.streamlit.io → *New app* → pilih repo, file utama `app.py`.
3. *Advanced settings → Secrets/Environment*: `OASIS_PUBLIC_DEMO = "1"` (memunculkan peringatan & memblokir status
   “Asli-internal”). Opsional LLM: `ANTHROPIC_API_KEY` di *Secrets* (tambahkan `anthropic` ke requirements).

**Hugging Face Spaces (Docker)** — buat Space tipe *Docker*, unggah repo; `Dockerfile` sudah menyetel
`OASIS_PUBLIC_DEMO=1`, membangkitkan data demo, dan membuka port 8501 (set `app_port: 8501` pada README Space).

**Docker lokal/intranet**
```bash
docker build -t oasis-poc .
docker run -p 8501:8501 oasis-poc
# untuk intranet dengan data internal: jalankan TANPA OASIS_PUBLIC_DEMO dan mount folder data privat
docker run -p 8501:8501 -e OASIS_PUBLIC_DEMO=0 -e OASIS_LOCAL_WORKBOOK=/data/wb.xlsx -v /path/privat:/data oasis-poc
```

**LLM opsional**: `pip install anthropic`, set `ANTHROPIC_API_KEY` (dan opsional `OASIS_LLM_MODEL`, default
`claude-sonnet-5-5`). Toggle muncul di sidebar. Yang dikirim hanya teks pertanyaan + daftar id/label katalog.

---

## 6. Temuan pada data statistik publik (sumber utama, otomatis oleh OASIS)

* **0 temuan kritis**: 8/8 sheet ada, Periode valid (66 bulan, Jan 2021–Jun 2026), tanpa duplikat/bulan hilang.
* **Satuan**: kamus kini mencantumkan `Rp M` (45 field), `%` (38, termasuk RBC), `Rp triliun` (2) → tidak ada lagi field
  “satuan belum terkonfirmasi”. `Rp M` dan `Rp triliun` tetap tidak dikonversi tanpa aturan eksplisit.
* **Pemetaan field**: 4 field per sheet asuransi (Tagihan Klaim/Premi …) hanya dapat dipetakan berdasarkan posisi kolom →
  *perlu konfirmasi*; field lain cocok persis/setelah normalisasi.
* **Uji aturan eksplisit** (DPK, LDR, Total Aset asuransi, FAR Pembiayaan, komponen investasi Dana Pensiun): lolos.
* **RBC Asuransi Umum identik dengan RBC Reasuransi** pada 66 periode → kemungkinan salah salin.
* **Pola kumulatif tahun berjalan** (heuristik) pada Laba Bank Umum, BPR, Pembiayaan, serta ROI R & ROI R+U Dana Pensiun.

## 6a. Desain antarmuka

* Favicon & logo bertema statistik (`assets/favicon.png`, SVG inline di header/sidebar).
* Header minimalis dengan grid halus, sparkline dekoratif, dan ringkasan cakupan data (indikator, subsektor, periode, kualitas).
* Kotak **“Tanyakan pada OASIS”** sebagai titik interaksi utama; contoh pertanyaan berupa **dropdown berkategori** yang
  otomatis mengisi kolom input.
* Urutan hasil: panel pemahaman pertanyaan → **Visualisasi & Perlu Perhatian** → Angka kunci → rincian metodologis →
  Executive summary & asal data. Tata letak responsif (kolom menumpuk di layar sempit; sidebar otomatis tertutup di ponsel).

## 7. Batasan POC

* Parser aturan mencakup pola pertanyaan umum (statistik, MoM/YoY, tren, perbandingan, korelasi, proyeksi, perhatian);
  kalimat sangat bebas dapat memerlukan LLM opsional atau penulisan ulang.
* Arah “memburuk” per indikator (config) dan threshold bersifat konfigurasi demo, bukan penilaian pengawasan.
* p-value tren dan korelasi tidak mengoreksi autokorelasi; dinyatakan indikatif.
* Data agregat subsektor bulanan; tidak ada drill-down per lembaga.
