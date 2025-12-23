import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. KONFIGURASI TEMA & WARNA
# ==========================================
CHART_PALETTE = {
    "moss_green": "#76944C", 
    "light_sage": "#C8DAA6", 
    "cream": "#FBF5DB",
    "honey_yellow": "#FFD21F", 
    "warm_grey": "#C0B6AC", 
    "dark_text": "#2F3632",
}
QUAL_COLORS = {"Premium": "#76944C", "Medium": "#FFD21F", "Pecah": "#C0B6AC"}
MONTHS_ID = ["Januari","Februari","Maret","April","Mei","Juni","Juli","Agustus","September","Oktober","November","Desember"]
MONTH_CAT = pd.CategoricalDtype(categories=MONTHS_ID, ordered=True)

# ==========================================
# 2. FUNGSI HELPER STATISTIK
# ==========================================
def keputusan_normal(p, alpha=0.05): return "NORMAL" if p >= alpha else "TIDAK NORMAL"
def keputusan_homogen(p, alpha=0.05): return "HOMOGEN" if p >= alpha else "TIDAK HOMOGEN"

def parse_harga_beras(file_like):
    raw = pd.read_csv(file_like, header=None)
    # Mencari index baris bulan
    idx_bulan = raw.apply(lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(), axis=1).idxmax()
    month_map = {j: str(raw.iloc[idx_bulan, j]).strip() for j in range(1, 13) if str(raw.iloc[idx_bulan, j]).strip() in MONTHS_ID}
    
    # Mencari Tahun
    tahun_raw = raw.iloc[max(0, idx_bulan-3):idx_bulan+1, :].astype(str)
    tahun = 2024
    for val in tahun_raw.values.flatten():
        if val.isdigit() and len(val) == 4:
            tahun = int(val)
            break

    records = []
    for i in range(idx_bulan+1, raw.shape[0]):
        qual = str(raw.iloc[i, 0]).strip().title()
        if qual not in ["Premium", "Medium", "Pecah"]: continue
        for col_idx, month in month_map.items():
            val = str(raw.iloc[i, col_idx]).strip().replace(",", "")
            try: records.append({"kualitas": qual, "bulan": month, "harga": float(val)})
            except: continue
            
    df = pd.DataFrame(records)
    df["bulan"] = df["bulan"].astype(MONTH_CAT)
    return df.sort_values(["bulan", "kualitas"]).reset_index(drop=True), tahun

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.set_page_config(page_title="Analisis Harga Beras RCBD", layout="wide")

st.markdown(f"""
<style>
    html, body, [class*="st-"], .stMarkdown, .stTable, .stDataFrame {{ font-family: "Times New Roman", Times, serif !important; }}
    .stApp {{ background-color: {CHART_PALETTE['cream']}; color: {CHART_PALETTE['dark_text']}; }}
    .header-box {{ background: linear-gradient(135deg, {CHART_PALETTE['moss_green']} 0%, {CHART_PALETTE['light_sage']} 100%); padding: 2.5rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem; }}
    code {{ color: #d63384; }}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1>Dashboard Analisis Harga Beras</h1><p>Laporan Statistik Formal: Randomized Complete Block Design (RCBD)</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📄 Pengaturan Data")
    up = st.file_uploader("Unggah File CSV (BPS)", type=["csv"])

if up:
    long, tahun = parse_harga_beras(io.BytesIO(up.getvalue()))
    wide = long.pivot_table(index="bulan", columns="kualitas", values="harga")
    
    t1, t2, t3, t4 = st.tabs(["📋 Data Check", "📊 EDA Deskriptif", "🧪 Uji Asumsi", "🔬 Hasil ANOVA & Post-Hoc"])

    # --- TAB 1: DATA CHECK ---
    with t1:
        st.markdown("### === DATA CHECK ===")
        c1, c2 = st.columns(2)
        with c1:
            st.text(f"Bulan: {MONTHS_ID}")
            st.text(f"Wide shape: {wide.shape} | Long shape: {long.shape}")
            st.text(f"Kualitas: {sorted(long['kualitas'].unique().tolist())}")
        with c2:
            st.text("Jumlah data per kualitas:")
            st.json(long.groupby("kualitas")["harga"].size().to_dict())
        st.dataframe(long.head(12), use_container_width=True)

    # --- TAB 2: EDA ---
    with t2:
        st.markdown("### === EDA (Deskriptif harga) ===")
        desc = long.groupby("kualitas")["harga"].agg(["count", "mean", "std", "min", "max"])
        st.table(desc.style.format("{:.2f}"))
        st.plotly_chart(px.line(long, x="bulan", y="harga", color="kualitas", color_discrete_map=QUAL_COLORS, markers=True), use_container_width=True)

    # --- TAB 3: UJI ASUMSI ---
    with t3:
        st.markdown("### === NORMALITAS DATA (Shapiro) ===")
        for k, g in long.groupby("kualitas"):
            p = stats.shapiro(g["harga"]).pvalue
            st.write(f"**{k:8s}** | p-value: `{p:.6f}` | Status: `{keputusan_normal(p)}`")

        model = ols('harga ~ C(kualitas) + C(bulan)', data=long).fit()
        resid = model.get_influence().resid_studentized_internal
        
        st.markdown("### === HOMOGENITAS & RESIDUAL ===")
        p_lev = stats.levene(*[g["harga"].to_numpy() for _, g in long.groupby("kualitas")]).pvalue
        st.write(f"Levene p-value: `{p_lev:.6f}` | `{keputusan_homogen(p_lev)}`")
        
        p_sh = stats.shapiro(resid).pvalue
        st.write(f"Shapiro Residual p-value: `{p_sh:.6f}` | `{keputusan_normal(p_sh)}`")

    # --- TAB 4: ANOVA & POST-HOC ---
    with t4:
        st.markdown("### === ANOVA (RCBD) ===")
        aov_table = anova_lm(model, typ=2)
        st.table(aov_table)

        # Effect Size
        ss_res = aov_table.loc["Residual", "sum_sq"]
        eta_q = aov_table.loc["C(kualitas)", "sum_sq"] / (aov_table.loc["C(kualitas)", "sum_sq"] + ss_res)
        st.write(f"**Partial Eta Squared (Kualitas):** `{eta_q:.4f}`")

        st.markdown("### === POST-HOC (Holm) ===")
        # Perbaikan Logika Post-Hoc menggunakan t_test model
        perbandingan = [
            ("Pecah - Medium", "C(kualitas)[T.Pecah] = 0"),
            ("Premium - Medium", "C(kualitas)[T.Premium] = 0"),
            ("Premium - Pecah", "C(kualitas)[T.Premium] - C(kualitas)[T.Pecah] = 0")
        ]
        
        ph_list = []
        raw_p = []
        for label, hypothesis in perbandingan:
            t_test = model.t_test(hypothesis)
            diff = float(t_test.effect)
            t_val = float(t_test.tvalue)
            p_val = float(t_test.pvalue)
            ph_list.append({"Pasangan": label, "diff": diff, "t": t_val, "p": p_val})
            raw_p.append(p_val)

        rej, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method='holm')
        for i, res in enumerate(ph_list):
            res["p_adj"] = p_adj[i]
            res["signif"] = rej[i]
            
        st.table(pd.DataFrame(ph_list).style.format({"diff": "{:.4f}", "t": "{:.3f}", "p": "{:.4e}", "p_adj": "{:.4e}"}))

else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
