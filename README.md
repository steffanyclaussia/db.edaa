# Dashboard Harga Beras (Streamlit)

Dashboard Streamlit untuk analisis harga beras 3 kualitas: **Premium, Medium, Pecah**.

## Cara menjalankan
1) Extract zip ini
2) Install dependency:
```bash
pip install -r requirements.txt
```
3) Jalankan:
```bash
streamlit run app.py
```

## Data
- Default: `data/Data_HargaBeras.csv` (contoh bawaan)
- Kamu juga bisa upload CSV lewat sidebar (format BPS / header bertingkat).

Dashboard akan otomatis merapikan data menjadi format LONG (Bulan, Kualitas, Harga), lalu menampilkan:
- Ringkasan & statistik deskriptif
- Visualisasi tren/barchart/boxplot
- **Repeated-Measures ANOVA (bulan sebagai blok)** + Friedman test + post-hoc paired t-test (Holm)
- Download data yang sudah rapi (LONG & WIDE)

## Palet warna
- #76944C, #C8DAA6, #FBF5DB, #FFD21F, #C0B6AC
Theme diset lewat `.streamlit/config.toml`.
