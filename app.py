import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# ==========================================
# 1. KONFIGURASI TEMA & PALET (USER THEME)
# ==========================================
PALETTE = {
    "green": "#76944C",
    "light_green": "#C8DAA6",
    "cream": "#FBF5DB",
    "yellow": "#FFD21F",
    "grey": "#C0B6AC",
    "text": "#2B2B2B",
    "white": "#FFFFFF"
}

QUAL_ORDER = ["Premium", "Medium", "Pecah"]
QUAL_COLORS = {
    "Premium": PALETTE["green"],
    "Medium": PALETTE["yellow"],
    "Pecah": PALETTE["grey"],
}

MONTHS_ID = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. LOGIKA PARSING & ANALISIS (BACKEND)
# ==========================================
def _read_raw_csv(file_like) -> pd.DataFrame:
    return pd.read_csv(file_like, header=None)

def parse_harga_beras(file_like) -> tuple[pd.DataFrame, dict]:
    raw = _read_raw_csv(file_like)
    month_row_idx = None
    jan_pos = None
    for i in range(raw.shape[0]):
        row = raw.iloc[i].astype(str)
        for j in range(raw.shape[1]):
            if row.iloc[j].strip() == "Januari":
                month_row_idx = i
                jan_pos = j
                break
        if month_row_idx is not None: break

    if month_row_idx is None:
        raise ValueError("Format tidak dikenali: tidak menemukan header bulan 'Januari'.")

    month_map = {}
    for j in range(jan_pos, raw.shape[1]):
        val = str(raw.iloc[month_row_idx, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val
        elif val.lower() in ["tahunan", "tahun", "annual", "nan", "none", ""]:
            if "Desember" in month_map.values(): break
            else: continue

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
        qual = str(qual).strip().title()
        if qual not in QUAL_ORDER: continue

        for col_idx, month in month_map.items():
            val = raw.iloc[i, col_idx]
            if pd.isna(val): continue
            s = str(val).strip().replace(",", "")
            try: harga = float(s)
            except ValueError: continue
            records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual, "Harga": harga})

    long_df = pd.DataFrame(records)
    long_df["Bulan"] = long_df["Bulan"].astype(MONTH_CAT)
    long_df = long_df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True)
    
    meta = {"tahun": tahun, "source": "Internal/Uploaded"}
    return long_df, meta

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean")
    return wide.reindex(columns=[q for q in QUAL_ORDER if q in wide.columns])

def repeated_measures_anova(long_df: pd.DataFrame):
    wide = make_wide(long_df).dropna()
    long_complete = wide.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
    aov = AnovaRM(long_complete, depvar="Harga", subject="Bulan", within=["Kualitas"]).fit()
    table = aov.anova_table
    f_val, p_val = table["F Value"].iloc[0], table["Pr > F"].iloc[0]
    df1, df2 = table["Num DF"].iloc[0], table["Den DF"].iloc[0]
    eta = (f_val * df1) / (f_val * df1 + df2)
    return {"table": table, "F": f_val, "p": p_val, "eta": eta, "wide": wide}

def paired_posthoc(wide: pd.DataFrame):
    import itertools
    pairs = list(itertools.combinations(wide.columns, 2))
    rows = []
    for a, b in pairs:
        x, y = wide[a].dropna(), wide[b].dropna()
        t, p = stats.ttest_rel(x, y)
        rows.append({"Pasangan": f"{a} vs {b}", "t": t, "p_raw": p, "Diff": x.mean() - y.mean()})
    df = pd.DataFrame(rows)
    if not df.empty:
        # Simple Holm-Bonferroni manual adjust
        df = df.sort_values("p_raw")
        df["p_holm"] = (df["p_raw"] * np.arange(len(df), 0, -1)).clip(upper=1)
    return df

# ==========================================
# 3. ANTARMUKA PENGGUNA (UI/UX)
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS untuk mempercantik tampilan ---
st.markdown(f"""
<style>
    .stApp {{ background-color: {PALETTE["cream"]}; }}
    [data-testid="stMetric"] {{
        background-color: {PALETTE["white"]};
        border: 1px solid {PALETTE["grey"]};
        border-radius: 10px;
        padding: 15px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {PALETTE["white"]};
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1601/1601730.png", width=80)
    st.title("Pengaturan Data")
    up = st.file_uploader("Upload CSV BPS", type=["csv"])
    use_sample = st.checkbox("Gunakan Data Contoh", value=(up is None))
    st.divider()
    mode = st.radio("Metode Statistik", ["Repeated-Measures ANOVA", "One-Way ANOVA (Independen)"])

# --- Data Loading ---
try:
    if up:
        long_df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
    else:
        # Fallback data dummy jika file tidak ada
        long_df, meta = parse_harga_beras("data/Data_HargaBeras.csv")
    wide = make_wide(long_df)
except Exception as e:
    st.error(f"Sistem membutuhkan file CSV yang valid. Pesan: {e}")
    st.stop()

# --- Top Header & Metrics ---
st.title("🌾 Dashboard Analisis Harga Beras")
st.caption(f"Tahun: {meta['tahun']} | Sumber: {meta['source']}")

m1, m2, m3, m4 = st.columns(4)
for i, qual in enumerate(QUAL_ORDER):
    if qual in wide.columns:
        latest = wide[qual].iloc[-1]
        first = wide[qual].iloc[0]
        diff = latest - first
        cols = [m1, m2, m3]
        cols[i].metric(qual, f"Rp {latest:,.0f}", f"{diff:,.0f} (YTD)", delta_color="inverse")
m4.metric("Total Data", f"{len(long_df)} Baris")

st.divider()

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Visualisasi Tren", "🔬 Uji Statistik", "📋 Tabel Data"])

with tab1:
    # 1. Line Chart
    st.subheader("Tren Harga Bulanan")
    fig_line = px.line(long_df, x="Bulan", y="Harga", color="Kualitas", 
                       color_discrete_map=QUAL_COLORS, markers=True, line_shape="spline")
    fig_line.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Boxplot & Bar
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Harga")
        fig_box = px.box(long_df, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS)
        st.plotly_chart(fig_box, use_container_width=True)
    with c2:
        st.subheader("Rata-rata Tahunan")
        mean_vals = long_df.groupby("Kualitas")["Harga"].mean().reset_index()
        fig_bar = px.bar(mean_vals, x="Kualitas", y="Harga", color="Kualitas", color_discrete_map=QUAL_COLORS, text_auto=".0f")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("Analisis Varians (ANOVA)")
    if mode == "Repeated-Measures ANOVA":
        res = repeated_measures_anova(long_df)
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("F-Statistic", f"{res['F']:.2f}")
        col_s2.metric("P-Value", f"{res['p']:.4g}")
        col_s3.metric("Effect Size (η²)", f"{res['eta']:.2f}")
        
        if res['p'] < 0.05:
            st.success("✅ Terdapat perbedaan harga yang signifikan antar kualitas beras (p < 0.05).")
        else:
            st.warning("⚠️ Tidak ditemukan perbedaan harga yang signifikan antar kualitas (p > 0.05).")

        with st.expander("Tabel Hasil Post-Hoc (Perbandingan Berpasangan)"):
            post = paired_posthoc(res["wide"])
            st.dataframe(post.style.format(precision=4).background_gradient(subset=['p_holm'], cmap='RdYlGn_r'), use_container_width=True)
    else:
        st.info("Mode One-Way ANOVA terpilih. Gunakan ini hanya jika data antar bulan dianggap tidak berhubungan.")

with tab3:
    st.subheader("Data Wide (Bulan x Kualitas)")
    st.dataframe(wide.style.format("{:,.0f}").highlight_max(axis=1, color="#FFD21F").highlight_min(axis=1, color="#C8DAA6"), use_container_width=True)
    
    st.subheader("Data Long (Format Analisis)")
    st.dataframe(long_df, use_container_width=True)
    
    csv = long_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Data Hasil Olahan", csv, "data_beras_bersih.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown("<center>Laporan Analisis Harga Beras - Dibuat dengan Streamlit & Plotly</center>", unsafe_allow_html=True)
