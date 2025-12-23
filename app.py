import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Import library untuk ANOVA RCBD (OLS Model)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ==========================================
# 1. KONFIGURASI TEMA & WARNA (PALETTE SESUAI GAMBAR)
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C",    # Hijau Daun (Utama)
    "light_sage": "#C8DAA6",    # Hijau Muda
    "cream": "#FBF5DB",         # Krem (Background)
    "honey_yellow": "#FFD21F",  # Kuning (Aksen)
    "warm_grey": "#C0B6AC",     # Abu-abu Hangat
    "dark_text": "#2F3632",     # Teks Gelap
}

QUAL_ORDER = ["Premium", "Medium", "Pecah"]
QUAL_COLORS = {
    "Premium": "#76944C",      
    "Medium": "#FFD21F",       
    "Pecah": "#C0B6AC",        
}

MONTHS_ID = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. FUNGSI PEMROSESAN DATA
# ==========================================
def parse_harga_beras(file_like) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(file_like, header=None)
    month_row_idx = None
    jan_pos = None
    for i in range(raw.shape[0]):
        row = raw.iloc[i].astype(str)
        for j in range(raw.shape[1]):
            if row.iloc[j].strip() == "Januari":
                month_row_idx, jan_pos = i, j
                break
        if month_row_idx is not None: break

    if month_row_idx is None:
        raise ValueError("Format BPS tidak ditemukan.")

    month_map = {j: str(raw.iloc[month_row_idx, j]).strip() for j in range(jan_pos, raw.shape[1]) 
                 if str(raw.iloc[month_row_idx, j]).strip() in MONTHS_ID}
    
    tahun = next((int(str(raw.iloc[r, c]).strip()) for r in range(max(0, month_row_idx-3), month_row_idx+1) 
                  for c in range(raw.shape[1]) if str(raw.iloc[r, c]).strip().isdigit() and len(str(raw.iloc[r, c]).strip()) == 4), None)

    records = []
    for i in range(month_row_idx+1, raw.shape[0]):
        qual = str(raw.iloc[i, 0]).strip()
        if not qual or qual.lower() == "nan": continue
        for col_idx, month in month_map.items():
            val = str(raw.iloc[i, col_idx]).strip().replace(",", "")
            try:
                records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual.title(), "Harga": float(val)})
            except: continue

    df = pd.DataFrame(records)
    df = df[df["Kualitas"].isin(QUAL_ORDER)].copy()
    df["Bulan"] = df["Bulan"].astype("string").astype(MONTH_CAT)
    return df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True), {"tahun": tahun}

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    return long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean").reindex(columns=QUAL_ORDER)

# ==========================================
# 3. STREAMLIT UI (TIMES NEW ROMAN & CLEAN DESIGN)
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

# CSS Injection untuk Font dan Perapihan Tampilan
st.markdown(f"""
<style>
    /* Mengubah seluruh font ke Times New Roman */
    html, body, [class*="st-"] {{
        font-family: "Times New Roman", Times, serif !important;
    }}

    .stApp {{ 
        background-color: {CHART_PALETTE['cream']}; 
        color: {CHART_PALETTE['dark_text']}; 
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {CHART_PALETTE['moss_green']};
        padding-top: 2rem;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
        font-family: "Times New Roman", Times, serif !important;
    }}

    /* Header Styling */
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2.5rem; 
        border-radius: 20px; 
        color: white; 
        text-align: center; 
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    
    /* Card Metrik */
    div[data-testid="stMetric"] {{
        background: white; 
        border-radius: 15px; 
        padding: 20px; 
        border-bottom: 6px solid {CHART_PALETTE['honey_yellow']};
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{ 
        gap: 15px; 
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white; 
        border-radius: 10px 10px 0 0; 
        color: {CHART_PALETTE['moss_green']};
        padding: 10px 30px;
        font-weight: bold;
        border: 1px solid #ddd;
    }}
    .stTabs [aria-selected="true"] {{ 
        background-color: {CHART_PALETTE['moss_green']} !important; 
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown(f"""
    <div class="header-box">
        <h1 style="font-family: 'Times New Roman', serif; font-size: 3rem; margin:0;">Dashboard Analisis Harga Beras</h1>
        <hr style="border: 1px solid rgba(255,255,255,0.3); width: 50%; margin: 10px auto;">
        <p style="font-size: 1.3rem; font-style: italic;">Metode Statistik: Randomized Complete Block Design (RCBD)</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Pengaturan Data")
    up = st.file_uploader("Unggah File CSV (BPS)", type=["csv"])
    st.divider()
    st.markdown("### 💡 Informasi")
    st.caption("Dashboard ini menganalisis pengaruh kualitas terhadap harga dengan mempertimbangkan variasi bulanan sebagai blok (kelompok).")

if up:
    df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide_df = make_wide(df)

    # Ringkasan Metrik Informatif
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Periode Data", f"Tahun {meta['tahun'] if meta['tahun'] else '-'}")
    with m2: st.metric("Rata-rata Premium", f"Rp {df[df['Kualitas']=='Premium']['Harga'].mean():,.0f}")
    with m3: st.metric("Rata-rata Medium", f"Rp {df[df['Kualitas']=='Medium']['Harga'].mean():,.0f}")
    with m4: st.metric("Rata-rata Pecah", f"Rp {df[df['Kualitas']=='Pecah']['Harga'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigasi Tab
    t1, t2, t3 = st.tabs(["📈 Tren Visual & Distribusi", "📋 Matriks Data Lengkap", "🔬 Analisis Statistik Mendalam"])

    with t1:
        # Grafik Tren dengan Font Times New Roman
        fig_line = px.line(df, x="Bulan", y="Harga", color="Kualitas", 
                           color_discrete_map=QUAL_COLORS, markers=True,
                           title="Visualisasi Tren Harga Bulanan")
        fig_line.update_layout(
            font_family="Times New Roman",
            plot_bgcolor='rgba(0,0,0,0.02)',
            xaxis_title="Bulan Pelaporan",
            yaxis_title="Harga (Rupiah)"
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df, x="Kualitas", y="Harga", color="Kualitas", 
                            color_discrete_map=QUAL_COLORS, title="Analisis Sebaran (Variabilitas) Harga")
            fig_box.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_box, use_container_width=True)
        with c2:
            avg_df = df.groupby("Kualitas")["Harga"].mean().reset_index()
            fig_bar = px.bar(avg_df, x="Kualitas", y="Harga", color="Kualitas", 
                            color_discrete_map=QUAL_COLORS, title="Perbandingan Harga Rata-rata Tahunan")
            fig_bar.update_layout(font_family="Times New Roman")
            st.plotly_chart(fig_bar, use_container_width=True)

    with t2:
        st.markdown("### 📊 Tabel Data Matriks")
        st.markdown("Tabel di bawah menampilkan ringkasan harga rata-rata per bulan untuk setiap kategori kualitas.")
        st.dataframe(wide_df.style.highlight_max(axis=1, color=CHART_PALETTE['light_sage']).format("{:,.0f}"), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Ekspor Data Hasil Pembersihan (CSV)",
            data=df.to_csv(index=False),
            file_name=f"analisis_beras_{meta['tahun']}.csv",
            mime="text/csv"
        )

    with t3:
        st.markdown("### 🔬 Uji Signifikansi ANOVA (RCBD)")
        st.info("RCBD digunakan untuk mengontrol variabel pengganggu (bulan) guna mendapatkan estimasi perbedaan harga antar kualitas yang lebih akurat.")
        
        long_complete = wide_df.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
        aov_table = anova_lm(model, typ=2)
        
        # Penamaan Tabel ANOVA
        aov_display = aov_table.rename(index={
            'C(Kualitas)': 'Kualitas (Perlakuan Utama)', 
            'C(Bulan)': 'Bulan (Faktor Kelompok/Blok)', 
            'Residual': 'Kesalahan Eksperimental (Error)'
        })
        
        st.table(aov_display.style.format("{:.4f}"))
        
        p_val = aov_table.loc["C(Kualitas)", "PR(>F)"]
        
        st.markdown("#### **📌 Kesimpulan Analisis:**")
        if p_val < 0.05:
            st.success(f"**SIGNIFIKAN**: Secara statistik, terdapat perbedaan harga yang nyata antar kategori kualitas beras (p-value = {p_val:.4f} < 0.05).")
        else:
            st.warning(f"**TIDAK SIGNIFIKAN**: Tidak ditemukan bukti statistik yang cukup untuk menyatakan adanya perbedaan harga antar kualitas pada tingkat kepercayaan 95% (p-value = {p_val:.4f}).")

else:
    # State Awal (Empty State)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.columns([1,2,1])[1].info("👋 Selamat Datang! Silakan unggah file CSV data harga beras melalui sidebar untuk memulai analisis.")
