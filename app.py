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
    "Premium": "#76944C",      # Moss Green
    "Medium": "#FFD21F",       # Honey Yellow
    "Pecah": "#C0B6AC",        # Warm Grey
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
        raise ValueError("Format BPS tidak ditemukan (Header Januari tidak ada).")

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
# 3. STREAMLIT UI (SINGLE PAGE FOCUS)
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    [data-testid="stSidebar"] {{ background-color: {CHART_PALETTE['moss_green']}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    .header-box {{
        background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
    }}
    div[data-testid="stMetric"] {{
        background: white; border-radius: 10px; padding: 15px; border-left: 5px solid {CHART_PALETTE['honey_yellow']};
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CHART_PALETTE['warm_grey']}; border-radius: 5px 5px 0 0; color: white;
    }}
    .stTabs [aria-selected="true"] {{ background-color: {CHART_PALETTE['moss_green']} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="header-box">
        <h1 style="margin:0;">Dashboard Analisis Harga Beras</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Metode Statistik: Rancangan Acak Kelompok (RCBD)</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Menu Data")
    up = st.file_uploader("Upload Data CSV BPS", type=["csv"])
    st.divider()
    st.write("**Catatan:**")
    st.caption("Dashboard ini secara otomatis menggunakan metode RCBD dengan Bulan sebagai variabel Blok untuk akurasi data runtun waktu.")

if up:
    df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide_df = make_wide(df)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tahun Analisis", meta['tahun'] if meta['tahun'] else "-")
    m2.metric("Rata-rata Premium", f"Rp {df[df['Kualitas']=='Premium']['Harga'].mean():,.0f}")
    m3.metric("Rata-rata Medium", f"Rp {df[df['Kualitas']=='Medium']['Harga'].mean():,.0f}")
    m4.metric("Rata-rata Pecah", f"Rp {df[df['Kualitas']=='Pecah']['Harga'].mean():,.0f}")

    t1, t2, t3 = st.tabs(["📊 Tren & Distribusi", "📋 Matriks Data", "🧮 Hasil Statistik RCBD"])

    with t1:
        st.plotly_chart(px.line(df, x="Bulan", y="Harga", color="Kualitas", 
                               color_discrete_map=QUAL_COLORS, markers=True,
                               title="Tren Harga Bulanan"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.box(df, x="Kualitas", y="Harga", color="Kualitas", 
                                  color_discrete_map=QUAL_COLORS, title="Sebaran Harga"), use_container_width=True)
        with c2:
            avg_df = df.groupby("Kualitas")["Harga"].mean().reset_index()
            st.plotly_chart(px.bar(avg_df, x="Kualitas", y="Harga", color="Kualitas", 
                                  color_discrete_map=QUAL_COLORS, title="Rata-rata Harga"), use_container_width=True)

    with t2:
        st.write("### Matriks Harga per Bulan")
        st.dataframe(wide_df.style.highlight_max(axis=1, color=CHART_PALETTE['light_sage']).format("{:,.0f}"))
        st.download_button("Download Data Cleaned", df.to_csv(index=False), "data_beras_bersih.csv", "text/csv")

    with t3:
        st.subheader("Tabel ANOVA (RCBD)")
        long_complete = wide_df.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
        aov_table = anova_lm(model, typ=2)
        
        # Penamaan tabel agar lebih informatif
        aov_display = aov_table.rename(index={'C(Kualitas)': 'Kualitas (Perlakuan)', 'C(Bulan)': 'Bulan (Blok)', 'Residual': 'Error'})
        st.table(aov_display.style.format("{:.4f}"))
        
        p_val = aov_table.loc["C(Kualitas)", "PR(>F)"]
        if p_val < 0.05:
            st.success(f"Kesimpulan: Terdapat perbedaan harga yang **SIGNIFIKAN** antar kategori kualitas (p-value: {p_val:.4f}).")
        else:
            st.warning(f"Kesimpulan: Tidak ditemukan perbedaan harga yang signifikan secara statistik (p-value: {p_val:.4f}).")

else:
    st.info("Silakan unggah file CSV BPS untuk mengaktifkan dashboard.")
