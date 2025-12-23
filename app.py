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
# 1. KONFIGURASI TEMA & WARNA (THEME FOMO)
# ==========================================
FOMO_PALETTE = {
    "maroon_dark": "#590d22",   # Background Sidebar
    "maroon": "#800f2f",        # Elemen Utama / Judul
    "red": "#c9184a",           # Header Gradient End
    "pink_dark": "#ff4d6d",     # Aksen Grafik
    "pink_light": "#ff758f",    # Aksen Grafik 2
    "soft_bg": "#fff0f3",       # Background Halaman Utama
    "white": "#ffffff",
    "text": "#2B2B2B",
}

# Urutan Kualitas & Mapping Warna
QUAL_ORDER = ["Premium", "Medium", "Pecah"]
QUAL_COLORS = {
    "Premium": "#800f2f",  # Merah Gelap
    "Medium": "#c9184a",   # Merah Cerah
    "Pecah": "#ffb3c1",    # Pink Muda
}

MONTHS_ID = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)


# ==========================================
# 2. FUNGSI PEMROSESAN DATA (HELPER)
# ==========================================
def _read_raw_csv(file_like) -> pd.DataFrame:
    return pd.read_csv(file_like, header=None)

def parse_harga_beras(file_like) -> tuple[pd.DataFrame, dict]:
    raw = _read_raw_csv(file_like)

    # Cari baris yang mengandung 'Januari'
    month_row_idx = None
    jan_pos = None
    for i in range(raw.shape[0]):
        row = raw.iloc[i].astype(str)
        for j in range(raw.shape[1]):
            if row.iloc[j].strip() == "Januari":
                month_row_idx = i
                jan_pos = j
                break
        if month_row_idx is not None:
            break

    if month_row_idx is None:
        raise ValueError("Format tidak dikenali: Header bulan 'Januari' tidak ditemukan.")

    month_map = {}
    for j in range(jan_pos, raw.shape[1]):
        val = str(raw.iloc[month_row_idx, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val
        elif val.lower() in ["tahunan", "tahun", "annual", "nan", "none", ""]:
            if "Desember" in month_map.values():
                break
    
    tahun = None
    for r in range(max(0, month_row_idx-3), month_row_idx+1):
        for c in range(raw.shape[1]):
            cell = raw.iloc[r, c]
            if pd.isna(cell): continue
            s = str(cell).strip()
            if s.isdigit() and len(s) == 4:
                tahun = int(s)
                break
        if tahun is not None: break

    records = []
    for i in range(month_row_idx+1, raw.shape[0]):
        qual = raw.iloc[i, 0]
        if pd.isna(qual): continue
        qual = str(qual).strip()
        if qual == "" or qual.lower() in ["nan", "none"]: continue

        for col_idx, month in month_map.items():
            val = raw.iloc[i, col_idx]
            if pd.isna(val): continue
            s = str(val).strip().replace(",", "")
            try:
                harga = float(s)
                records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual, "Harga": harga})
            except ValueError:
                continue

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError("Tidak ada data numerik yang berhasil dibaca.")

    long_df["Kualitas"] = long_df["Kualitas"].str.strip().str.title()
    long_df = long_df[long_df["Kualitas"].isin(QUAL_ORDER)].copy()
    long_df["Bulan"] = long_df["Bulan"].astype("string").astype(MONTH_CAT)
    long_df = long_df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True)

    meta = {"tahun": tahun, "qualities_found": sorted(long_df["Kualitas"].unique().tolist())}
    return long_df, meta

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean")
    wide = wide.reindex(columns=[q for q in QUAL_ORDER if q in wide.columns])
    return wide

# ==========================================
# 3. FUNGSI STATISTIK
# ==========================================
def two_way_anova_rcbd(long_df: pd.DataFrame):
    wide = make_wide(long_df).dropna()
    long_complete = wide.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
    
    model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
    aov_table = anova_lm(model, typ=2)
    
    row_qual = aov_table.loc["C(Kualitas)"]
    ss_effect = row_qual["sum_sq"]
    ss_error = aov_table.loc["Residual", "sum_sq"]
    
    display_table = aov_table.rename(index={
        "C(Kualitas)": "Kualitas (Perlakuan)",
        "C(Bulan)": "Bulan (Blok)",
        "Residual": "Error / Sisaan"
    })

    return {
        "table": display_table, 
        "F": row_qual["F"], 
        "p": row_qual["PR(>F)"], 
        "partial_eta2": ss_effect / (ss_effect + ss_error),
        "wide": wide
    }

def paired_posthoc(wide: pd.DataFrame):
    pairs = [("Premium","Medium"), ("Premium","Pecah"), ("Medium","Pecah")]
    rows = []
    for a,b in pairs:
        if a in wide.columns and b in wide.columns:
            x, y = wide[a], wide[b]
            t, p = stats.ttest_rel(x, y)
            diff = x - y
            rows.append({
                "Pasangan": f"{a} vs {b}", 
                "t-stat": t, "p-value": p, 
                "Selisih Mean": np.mean(diff)
            })
    return pd.DataFrame(rows)

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Dashboard Harga Beras", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {FOMO_PALETTE['soft_bg']}; color: {FOMO_PALETTE['text']}; }}
    [data-testid="stSidebar"] {{ background-color: {FOMO_PALETTE['maroon_dark']}; }}
    .header-box {{
        background: linear-gradient(135deg, {FOMO_PALETTE['maroon_dark']} 0%, {FOMO_PALETTE['red']} 100%);
        padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1>Dashboard Harga Beras</h1><p>Analisis Statistik Kualitas Beras</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("📂 Panel Data")
    up = st.file_uploader("Upload CSV", type=["csv"])
    mode = st.radio("Metode:", ["RCBD (Two-Way ANOVA)", "One-Way ANOVA"])

# Load Data
if up:
    long_df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide = make_wide(long_df)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Tahun", meta['tahun'])
    c2.metric("Rata-rata Harga", f"Rp {long_df['Harga'].mean():,.0f}")
    c3.metric("Data Point", len(long_df))

    tab_vis, tab_data, tab_stat = st.tabs(["📊 Visualisasi", "📋 Data", "🧮 Statistik"])

    with tab_vis:
        fig = px.line(long_df, x="Bulan", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.box(long_df, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS), use_container_width=True)
        with col_b:
            avg_price = long_df.groupby("Kualitas")["Harga"].mean().reset_index()
            st.plotly_chart(px.bar(avg_price, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS), use_container_width=True)

    with tab_data:
        st.subheader("Tabel Matriks Harga")
        # PERBAIKAN: Menutup tanda kurung dan string yang sebelumnya error
        with st.expander("Klik untuk melihat Tabel Lengkap (Format Matriks)"):
            st.dataframe(wide.style.format("{:,.0f}"))
        st.subheader("Ringkasan Statistik")
        st.dataframe(wide.describe().T)

    with tab_stat:
        if "RCBD" in mode:
            res = two_way_anova_rcbd(long_df)
            st.write("### Tabel ANOVA (RCBD)")
            st.table(res["table"])
            if res["p"] < 0.05:
                st.success(f"Signifikan (p={res['p']:.4f}). Ada perbedaan harga antar kualitas.")
                st.write("### Uji Lanjut (Post-Hoc)")
                st.dataframe(paired_posthoc(res["wide"]))
            else:
                st.warning("Tidak ada perbedaan signifikan.")
        else:
            st.info("Fitur One-Way ANOVA terpilih.")

else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
