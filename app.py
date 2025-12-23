import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# =========================
# Theme palette (from user)
# =========================
PALETTE = {
    "green": "#76944C",
    "light_green": "#C8DAA6",
    "cream": "#FBF5DB",
    "yellow": "#FFD21F",
    "grey": "#C0B6AC",
    "text": "#2B2B2B",
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

# -------------------------
# Helpers
# -------------------------
def _read_raw_csv(file_like) -> pd.DataFrame:
    # Read as raw matrix to handle "multi-row headers"
    return pd.read_csv(file_like, header=None)

def parse_harga_beras(file_like) -> tuple[pd.DataFrame, dict]:
    """
    Parse the uploaded 'Data_HargaBeras.csv' format:
    - A month header row containing 'Januari'..'Desember'
    - Rows below: quality labels in col 0 and monthly values across columns
    Returns:
        long_df: columns [Tahun, Bulan, Kualitas, Harga]
        meta: dict with parsed notes
    """
    raw = _read_raw_csv(file_like)

    # Find row index that contains 'Januari'
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
        raise ValueError("Format tidak dikenali: tidak menemukan header bulan 'Januari'.")

    # Extract month columns mapping col_idx -> month_name until Desember
    month_map = {}
    for j in range(jan_pos, raw.shape[1]):
        val = str(raw.iloc[month_row_idx, j]).strip()
        if val in MONTHS_ID:
            month_map[j] = val
        elif val.lower() in ["tahunan", "tahun", "annual", "nan", "none", ""]:
            # stop if we've already collected Desember; else continue
            if "Desember" in month_map.values():
                break
            else:
                continue
        else:
            # Keep going (some files may have filler columns)
            continue

    # Try parse year from a nearby row (often one row above month row)
    tahun = None
    for r in range(max(0, month_row_idx-3), month_row_idx+1):
        for c in range(raw.shape[1]):
            cell = raw.iloc[r, c]
            if pd.isna(cell):
                continue
            s = str(cell).strip()
            if s.isdigit() and len(s) == 4:
                tahun = int(s)
                break
        if tahun is not None:
            break

    # Data rows start after month_row_idx
    records = []
    for i in range(month_row_idx+1, raw.shape[0]):
        qual = raw.iloc[i, 0]
        if pd.isna(qual):
            continue
        qual = str(qual).strip()
        if qual == "" or qual.lower() in ["nan", "none"]:
            continue

        for col_idx, month in month_map.items():
            val = raw.iloc[i, col_idx]
            if pd.isna(val):
                continue
            # Coerce numeric: handle commas, dashes, etc.
            s = str(val).strip().replace(",", "")
            if s in ["-", "—", "–", ""]:
                continue
            try:
                harga = float(s)
            except ValueError:
                continue

            records.append({"Tahun": tahun, "Bulan": month, "Kualitas": qual, "Harga": harga})

    long_df = pd.DataFrame(records)
    if long_df.empty:
        raise ValueError("Tidak ada data numerik yang berhasil diparse dari file.")

    # Clean/standardize names
    long_df["Kualitas"] = long_df["Kualitas"].str.strip().str.title()
    # Fix likely variants
    long_df["Kualitas"] = long_df["Kualitas"].replace({
        "Pecah": "Pecah",
        "Medium": "Medium",
        "Premium": "Premium",
    })
    long_df = long_df[long_df["Kualitas"].isin(QUAL_ORDER)].copy()

    # Order months
    long_df["Bulan"] = long_df["Bulan"].astype("string")
    long_df["Bulan"] = long_df["Bulan"].astype(MONTH_CAT)

    # Sort
    long_df = long_df.sort_values(["Bulan", "Kualitas"]).reset_index(drop=True)

    meta = {
        "tahun": tahun,
        "month_row_idx": int(month_row_idx),
        "n_rows_long": int(long_df.shape[0]),
        "qualities_found": sorted(long_df["Kualitas"].unique().tolist()),
    }
    return long_df, meta

def make_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    wide = long_df.pivot_table(index="Bulan", columns="Kualitas", values="Harga", aggfunc="mean")
    # Ensure order
    wide = wide.reindex(columns=[q for q in QUAL_ORDER if q in wide.columns])
    wide = wide.sort_index()
    return wide

def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down)."""
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

def repeated_measures_anova(long_df: pd.DataFrame):
    # Subject/block = Bulan, within factor = Kualitas
    # Needs complete cases across qualities; pivot then back to long ensures that.
    wide = make_wide(long_df).dropna()
    long_complete = (
        wide.reset_index()
        .melt(id_vars=["Bulan"], value_vars=wide.columns, var_name="Kualitas", value_name="Harga")
        .dropna()
    )
    aov = AnovaRM(long_complete, depvar="Harga", subject="Bulan", within=["Kualitas"]).fit()
    table = aov.anova_table.copy()

    # Extract degrees of freedom
    df_effect = float(table["Num DF"].iloc[0])
    df_error = float(table["Den DF"].iloc[0])
    F = float(table["F Value"].iloc[0])
    p = float(table["Pr > F"].iloc[0])

    # Partial eta squared
    partial_eta2 = (F * df_effect) / (F * df_effect + df_error)

    return {"table": table, "F": F, "p": p, "df1": df_effect, "df2": df_error, "partial_eta2": partial_eta2, "wide": wide}

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
        dz = np.nanmean(diff) / np.nanstd(diff, ddof=1) if np.nanstd(diff, ddof=1) != 0 else np.nan
        rows.append({"Pasangan": f"{a} vs {b}", "t": t, "p_raw": p, "Mean Diff (a-b)": np.nanmean(diff), "Cohen dz": dz})
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Dashboard Harga Beras", layout="wide")

st.markdown(f"""
<style>
/* subtle card styling */
div[data-testid="stMetric"] {{
  background: white;
  border: 1px solid {PALETTE["grey"]};
  border-radius: 16px;
  padding: 14px;
}}
/* tighten top padding */
.block-container {{
  padding-top: 1.2rem;
}}
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Harga Beras (Premium • Medium • Pecah)")

with st.sidebar:
    st.header("Data")
    st.caption("Upload CSV format BPS (header bertingkat) atau pakai contoh bawaan.")
    up = st.file_uploader("Upload Data_HargaBeras.csv", type=["csv"])
    use_sample = st.toggle("Pakai data contoh bawaan", value=(up is None))
    st.divider()
    st.header("Analisis")
    mode = st.radio(
        "Metode uji varians",
        ["Recommended: Repeated-Measures (Bulan sebagai blok)", "Opsional: One-Way ANOVA (anggap independen)"],
        index=0
    )

# Load data
try:
    if up is not None and not use_sample:
        long_df, meta = parse_harga_beras(io.BytesIO(up.getvalue()))
        source_label = f"File upload: {up.name}"
    else:
        long_df, meta = parse_harga_beras("data/Data_HargaBeras.csv")
        source_label = "Data contoh: data/Data_HargaBeras.csv"
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

wide = make_wide(long_df)

# Top summary metrics
colA, colB, colC, colD = st.columns([1.2,1,1,1])
tahun_txt = str(meta["tahun"]) if meta.get("tahun") else "—"
colA.metric("Sumber", source_label)
colB.metric("Tahun", tahun_txt)
colC.metric("Jumlah bulan", int(wide.shape[0]))
colD.metric("Jumlah kualitas", int(wide.shape[1]))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Visualisasi", "Uji Varians", "Data & Unduh"])

with tab1:
    st.subheader("Tabel harga (bulan × kualitas)")
    st.dataframe(wide.style.format("{:,.2f}"), use_container_width=True)

    st.subheader("Statistik deskriptif per kualitas")
    desc = long_df.groupby("Kualitas")  ["Harga"].agg(["count","mean","std","min","max"]).reindex(QUAL_ORDER)
    st.dataframe(desc.style.format({"mean":"{:,.2f}","std":"{:,.2f}","min":"{:,.2f}","max":"{:,.2f}"}), use_container_width=True)

with tab2:
    st.subheader("Tren harga per bulan")
    plot_df = long_df.copy()
    plot_df["Bulan"] = plot_df["Bulan"].astype("string")
    fig = px.line(
        plot_df, x="Bulan", y="Harga", color="Kualitas",
        color_discrete_map=QUAL_COLORS, markers=True
    )
    fig.update_layout(legend_title_text="Kualitas", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rata-rata harga per kualitas")
        mean_df = (plot_df.groupby("Kualitas", as_index=False)["Harga"].mean()
                   .assign(Kualitas=lambda d: pd.Categorical(d["Kualitas"], QUAL_ORDER, ordered=True))
                   .sort_values("Kualitas"))
        fig2 = px.bar(mean_df, x="Kualitas", y="Harga", color="Kualitas",
                      color_discrete_map=QUAL_COLORS)
        fig2.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Sebaran harga (boxplot)")
        fig3 = px.box(plot_df, x="Kualitas", y="Harga", color="Kualitas",
                      category_orders={"Kualitas": QUAL_ORDER},
                      color_discrete_map=QUAL_COLORS, points="all")
        fig3.update_layout(showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Uji varians antar kualitas")
    st.caption("Karena data diukur per bulan untuk semua kualitas, metode yang paling tepat biasanya Repeated-Measures/Randomized Block (bulan = blok).")

    if mode.startswith("Recommended"):
        rm = repeated_measures_anova(long_df)
        st.markdown("### ✅ Repeated-Measures ANOVA (Bulan sebagai blok)")
        m1, m2, m3 = st.columns(3)
        m1.metric("F", f"{rm['F']:.3f}")
        m2.metric("p-value", f"{rm['p']:.3g}")
        m3.metric("Partial η²", f"{rm['partial_eta2']:.3f}")

        st.write("Tabel ANOVA:")
        st.dataframe(rm["table"], use_container_width=True)

        st.markdown("### Friedman test (nonparametrik)")
        fr_stat, fr_p, cols = friedman_test(rm["wide"].dropna())
        st.write(f"χ²({len(cols)-1}) = **{fr_stat:.3f}**, p = **{fr_p:.3g}**")

        st.markdown("### Post-hoc berpasangan (paired t-test + Holm)")
        post = paired_posthoc(rm["wide"].dropna())
        if not post.empty:
            post2 = post.copy()
            for c in ["t","p_raw","p_holm","Mean Diff (a-b)","Cohen dz"]:
                if c in post2.columns:
                    post2[c] = pd.to_numeric(post2[c], errors="coerce")
            st.dataframe(post2.style.format({
                "t":"{:.3f}",
                "p_raw":"{:.4f}",
                "p_holm":"{:.4f}",
                "Mean Diff (a-b)":"{:,.2f}",
                "Cohen dz":"{:.3f}"
            }), use_container_width=True)
        else:
            st.info("Data belum lengkap untuk post-hoc.")

    else:
        ow = one_way_anova_optional(long_df)
        st.markdown("### ⚠️ One-Way ANOVA (anggap tiap bulan independen)")
        st.warning("Metode ini mengasumsikan observasi antar bulan independen. Kalau data per bulan bergerak bareng, ini bisa kurang tepat.")
        m1, m2, m3 = st.columns(3)
        m1.metric("F", f"{ow['F']:.3f}")
        m2.metric("p-value", f"{ow['p']:.3g}")
        m3.metric("Levene p", f"{ow['levene_p']:.3g}")

        st.markdown("Normalitas (Shapiro-Wilk) per kualitas:")
        rows=[]
        for q,(stat,p) in ow["shapiro"].items():
            rows.append({"Kualitas": q, "W": stat, "p": p})
        st.dataframe(pd.DataFrame(rows).style.format({"W":"{:.3f}","p":"{:.4f}"}), use_container_width=True)

with tab4:
    st.subheader("Download data yang sudah rapi")
    long_out = long_df.copy()
    long_out["Bulan"] = long_out["Bulan"].astype("string")
    wide_out = wide.copy()
    wide_out.index = wide_out.index.astype("string")

    st.download_button(
        "⬇️ Download data LONG (CSV)",
        data=long_out.to_csv(index=False).encode("utf-8"),
        file_name="harga_beras_long.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download data WIDE (CSV)",
        data=wide_out.to_csv().encode("utf-8"),
        file_name="harga_beras_wide.csv",
        mime="text/csv",
    )

    st.subheader("Preview data LONG")
    st.dataframe(long_out, use_container_width=True)
