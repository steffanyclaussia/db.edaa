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

# Urutan Kualitas & Mapping Warna (Merah Gradasi)
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
    """
    Parsing format BPS yang memiliki header bertingkat tidak standar.
    """
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

    # Mapping kolom ke nama bulan
    month_map = {}
    for j in range(jan_pos, raw.shape[1]):
        val = str(raw.iloc[month_row_idx, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val
        elif val.lower() in ["tahunan", "tahun", "annual", "nan", "none", ""]:
            if "Desember" in month_map.values():
                break
            else:
                continue
        else:
            continue

    # Cari Tahun
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

    # Ambil data baris demi baris
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
            if s in ["-", "—", "–", ""]: continue
            try:
                harga = float(s)
            except ValueError:
                continue
            records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual, "Harga": harga})

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError("Tidak ada data numerik yang berhasil dibaca.")

    # Bersihkan nama kualitas
    long_df["Kualitas"] = long_df["Kualitas"].str.strip().str.title()
    long_df["Kualitas"] = long_df["Kualitas"].replace({
        "Pecah": "Pecah", "Medium": "Medium", "Premium": "Premium"
    })
    long_df = long_df[long_df["Kualitas"].isin(QUAL_ORDER)].copy()
    
    # Urutkan bulan
    long_df["Bulan"] = long_df["Bulan"].astype("string").astype(MONTH_CAT)
    long_df = long_df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True)

    meta = {
        "tahun": tahun,
        "month_row_idx": int(month_row_idx),
        "qualities_found": sorted(long_df["Kualitas"].unique().tolist()),
    }
    return long_df, meta

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean")
    wide = wide.reindex(columns=[q for q in QUAL_ORDER if q in wide.columns])
    wide = wide.sort_index()
    return wide

# ==========================================
# 3. FUNGSI STATISTIK (RCBD & POST-HOC)
# ==========================================

def two_way_anova_rcbd(long_df: pd.DataFrame):
    """
    Melakukan Two-Way ANOVA tanpa interaksi (RCBD / RAK).
    Model Linear: Harga ~ C(Kualitas) + C(Bulan)
    """
    # 1. Pastikan data balanced (lengkap)
    wide = make_wide(long_df).dropna()
    long_complete = (
        wide.reset_index()
        .melt(id_vars=["Bulan"], value_vars=wide.columns, var_name="Kualitas", value_name="Harga")
        .dropna()
    )
    
    # 2. Buat model OLS: Variabel Respon ~ Perlakuan + Blok
    # C() menandakan variabel kategorikal
    formula = 'Harga ~ C(Kualitas) + C(Bulan)'
    model = ols(formula, data=long_complete).fit()
    
    # 3. Buat Tabel ANOVA Type 2
    aov_table = anova_lm(model, typ=2)
    
    # 4. Ekstrak statistik untuk Kualitas (Perlakuan)
    row_qual = aov_table.loc["C(Kualitas)"]
    df_effect = row_qual["df"]
    df_error = aov_table.loc["Residual", "df"]
    F = row_qual["F"]
    p = row_qual["PR(>F)"]
    
    # Hitung Partial Eta Squared (Effect Size)
    ss_effect = row_qual["sum_sq"]
    ss_error = aov_table.loc["Residual", "sum_sq"]
    partial_eta2 = ss_effect / (ss_effect + ss_error)

    # Format ulang tabel untuk display
    display_table = aov_table[["sum_sq", "df", "F", "PR(>F)"]].copy()
    display_table.columns = ["Sum Sq", "DF", "F Value", "Pr > F"]
    display_table.index.name = "Source"
    
    # Ganti nama index agar lebih cantik
    index_map = {
        "C(Kualitas)": "Kualitas (Perlakuan)",
        "C(Bulan)": "Bulan (Blok)",
        "Residual": "Error / Sisaan"
    }
    display_table = display_table.rename(index=index_map)

    return {
        "table": display_table, 
        "F": F, 
        "p": p, 
        "partial_eta2": partial_eta2, 
        "wide": wide
    }

def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni correction"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        factor = m - k
        val = factor * pvals[idx]
        val = min(val, 1.0)
        prev = max(prev, val)
        adj[idx] = prev
    return adj.tolist()

def friedman_test(wide: pd.DataFrame):
    cols = [c for c in QUAL_ORDER if c in wide.columns]
    arrays = [wide[c].to_numpy() for c in cols]
    stat, p = stats.friedmanchisquare(*arrays)
    return stat, p, cols

def paired_posthoc(wide: pd.DataFrame):
    pairs = [("Premium","Medium"), ("Premium","Pecah"), ("Medium","Pecah")]
    rows = []
    raw_ps = []
    for a,b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        x = wide[a].to_numpy()
        y = wide[b].to_numpy()
        t, p = stats.ttest_rel(x, y, nan_policy="omit")
        diff = x - y
        # Cohen's dz
        sd_diff = np.nanstd(diff, ddof=1)
        dz = np.nanmean(diff) / sd_diff if sd_diff != 0 else np.nan
        
        rows.append({
            "Pasangan": f"{a} vs {b}", 
            "t-stat": t, 
            "p_raw": p, 
            "Selisih Mean": np.nanmean(diff), 
            "Cohen dz": dz
        })
        raw_ps.append(p)

    if rows:
        adj = holm_adjust(raw_ps)
        for r, p_adj in zip(rows, adj):
            r["p_holm"] = p_adj
    return pd.DataFrame(rows)

def one_way_anova_optional(long_df: pd.DataFrame):
    wide = make_wide(long_df).dropna()
    groups = [wide[q].to_numpy() for q in QUAL_ORDER if q in wide.columns]
    F, p = stats.f_oneway(*groups)
    lev_stat, lev_p = stats.levene(*groups)
    shaps = {}
    for q in QUAL_ORDER:
        if q in wide.columns:
            shaps[q] = stats.shapiro(wide[q].to_numpy())
    return {"F": F, "p": p, "levene_p": lev_p, "shapiro": shaps}

# ==========================================
# 4. STREAMLIT UI & CSS (VISUALISASI)
# ==========================================

st.set_page_config(page_title="Dashboard Harga Beras", layout="wide")

# --- CSS INJECTION (Gaya Fomo: Merah/Maroon/Pink) ---
st.markdown(f"""
<style>
    /* Background Utama */
    .stApp {{
        background-color: {FOMO_PALETTE['soft_bg']};
        color: {FOMO_PALETTE['text']};
    }}

    /* Sidebar - Maroon Dark */
    [data-testid="stSidebar"] {{
        background-color: {FOMO_PALETTE['maroon_dark']};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span {{
        color: white !important;
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {{
        color: #f0f0f0 !important;
    }}

    /* Header Banner Gradient */
    .header-box {{
        background: linear-gradient(135deg, {FOMO_PALETTE['maroon_dark']} 0%, {FOMO_PALETTE['red']} 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(128, 15, 47, 0.4);
    }}
    .header-box h1 {{
        color: white !important;
        margin: 0;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
    }}
    .header-box p {{
        color: #ffccd5;
        font-size: 1.1rem;
        margin-top: 5px;
    }}

    /* Metric Cards (Kotak Putih Shadow) */
    div[data-testid="stMetric"] {{
        background: white;
        border: none;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        text-align: center;
    }}
    div[data-testid="stMetric"] label {{
        color: {FOMO_PALETTE['maroon']};
        font-weight: 600;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {FOMO_PALETTE['text']};
        font-size: 24px;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #ddd;
        color: {FOMO_PALETTE['text']};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {FOMO_PALETTE['maroon']} !important;
        color: white !important;
    }}

    /* Subheader color */
    h3 {{
        color: {FOMO_PALETTE['maroon']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-box">
        <h1>Dashboard Harga Beras</h1>
        <p>Analisis Perbandingan Harga & Statistik: Premium • Medium • Pecah</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Panel Data")
    st.info("Upload file CSV format BPS atau gunakan data contoh.")
    
    up = st.file_uploader("Upload Data_HargaBeras.csv", type=["csv"])
    use_sample = st.toggle("Pakai Data Contoh", value=(up is None))
    
    st.divider()
    st.subheader("⚙️ Metode Analisis")
    mode = st.radio(
        "Pilih Uji Statistik:",
        ["Recommended: RCBD / RAK (Two-Way ANOVA)", "Opsional: One-Way ANOVA"],
        index=0
    )
    if mode.startswith("Recommended"):
        st.caption("✅ Menggunakan Bulan sebagai BLOK (Kelompok) untuk mengurangi error varians.")
    else:
        st.caption("⚠️ Mengasumsikan data tiap bulan independen (kurang disarankan untuk data runtun waktu).")

# --- LOAD DATA ---
try:
    if up is not None and not use_sample:
        long_df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
        source_label = f"File: {up.name}"
    else:
        # Coba load dari path lokal, jika error buat dummy data
        try:
            long_df, meta = parse_harga_beras("data/Data_HargaBeras.csv")
            source_label = "Data Sample: BPS"
        except FileNotFoundError:
            st.warning("Data sample tidak ditemukan. Silakan upload file CSV Anda.")
            st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca data: {e}")
    st.stop()

wide = make_wide(long_df)

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sumber Data", source_label)
col2.metric("Tahun Data", str(meta["tahun"]) if meta.get("tahun") else "—")
col3.metric("Jumlah Bulan", int(wide.shape[0]))
col4.metric("Kategori Beras", int(wide.shape[1]))

st.divider()

# --- TABS CONTENT ---
tab_vis, tab_data, tab_stat, tab_down = st.tabs([
    "📊 Visualisasi Grafik", 
    "📋 Ringkasan Data", 
    "🧮 Uji Statistik (ANOVA)", 
    "📥 Unduh Laporan"
])

# 1. TAB VISUALISASI
with tab_vis:
    st.subheader("Tren & Distribusi Harga")
    
    # Line Chart
    plot_df = long_df.copy()
    plot_df["Bulan"] = plot_df["Bulan"].astype("string")
    fig = px.line(
        plot_df, x="Bulan", y="Harga", color="Kualitas",
        color_discrete_map=QUAL_COLORS, markers=True,
        title="Pergerakan Harga Bulanan"
    )
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f2f2f2'),
        yaxis=dict(showgrid=True, gridcolor='#f2f2f2'),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Bar Chart Rata-rata
        mean_df = plot_df.groupby("Kualitas", as_index=False)["Harga"].mean()
        mean_df["Kualitas"] = pd.Categorical(mean_df["Kualitas"], QUAL_ORDER, ordered=True)
        mean_df = mean_df.sort_values("Kualitas")
        
        fig2 = px.bar(
            mean_df, x="Kualitas", y="Harga", color="Kualitas",
            color_discrete_map=QUAL_COLORS, text_auto='.2s',
            title="Rata-rata Harga per Tahun"
        )
        fig2.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        # Boxplot
        fig3 = px.box(
            plot_df, x="Kualitas", y="Harga", color="Kualitas",
            category_orders={"Kualitas": QUAL_ORDER},
            color_discrete_map=QUAL_COLORS, points="all",
            title="Sebaran (Distribusi) Harga"
        )
        fig3.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig3, use_container_width=True)

# 2. TAB DATA RINGKASAN
with tab_data:
    st.subheader("Tabel Data Harga")
    with st.expander("Klik untuk melihat Tabel Lengkap (Format
