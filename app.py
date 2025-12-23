import io
import pandas as pd
import numpy as np
import streamlit as st  # Tambahkan ini
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Konfigurasi Font untuk Streamlit agar rapi
plt.rcParams['font.family'] = 'serif'

# Judul Dashboard
st.title("Analisis Statistik Harga Beras")

# --- PARAMETER ---
ALPHA = 0.05
KEEP = ["premium", "medium", "pecah"]

def keputusan_normal(p, alpha=0.05):
    return "NORMAL" if p >= alpha else "TIDAK NORMAL"

def keputusan_homogen(p, alpha=0.05):
    return "HOMOGEN" if p >= alpha else "TIDAK HOMOGEN"

# --- 1) DATA LOADING ---
# Menggunakan file uploader agar lebih fleksibel di Streamlit
uploaded_file = st.sidebar.file_uploader("Unggah file Data_HargaBeras.csv", type=["csv"])

if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file, header=None)

    # Deteksi Baris Bulan
    idx_bulan = raw.apply(
        lambda r: r.astype(str).str.contains(r"\bJanuari\b", case=False, na=False).any(),
        axis=1
    ).idxmax()

    bulan = raw.loc[idx_bulan, 1:12].tolist()
    
    # Preprocessing
    wide = raw.loc[idx_bulan + 1:, [0] + list(range(1, 13))].copy()
    wide.columns = ["kualitas"] + bulan
    wide["kualitas"] = wide["kualitas"].astype(str).str.strip()
    wide = wide[wide["kualitas"].str.lower().isin(KEEP)].copy()

    for b in bulan:
        wide[b] = pd.to_numeric(wide[b], errors="coerce")

    long = wide.melt(
        id_vars="kualitas",
        value_vars=bulan,
        var_name="bulan",
        value_name="harga"
    ).dropna(subset=["harga"]).copy()

    long["bulan"] = pd.Categorical(long["bulan"], categories=bulan, ordered=True)
    long["kualitas"] = long["kualitas"].str.strip().str.title()
    long = long.sort_values(["bulan", "kualitas"]).reset_index(drop=True)

    # --- 2) EDA DISPLAY ---
    st.header("1. Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tren Harga")
        fig1, ax1 = plt.subplots()
        pivot = long.pivot_table(index="bulan", columns="kualitas", values="harga")
        pivot.plot(ax=ax1, marker='o')
        plt.xticks(rotation=45)
        st.pyplot(fig1) # Gunakan st.pyplot pengganti plt.show()

    with col2:
        st.subheader("Sebaran Harga (Boxplot)")
        fig2, ax2 = plt.subplots()
        kvals = sorted(long["kualitas"].unique())
        data_plot = [long.loc[long["kualitas"] == k, "harga"].values for k in kvals]
        ax2.boxplot(data_plot, labels=kvals)
        st.pyplot(fig2)

    # --- 3) UJI NORMALITAS & HOMOGENITAS ---
    st.header("2. Uji Asumsi ANOVA")
    
    # Normalitas Residual
    model = smf.ols("harga ~ C(kualitas) + C(bulan)", data=long).fit()
    infl = model.get_influence()
    resid_stud = infl.resid_studentized_internal
    p_sh = stats.shapiro(resid_stud).pvalue
    
    st.write(f"**Uji Shapiro-Wilk (Residual):** p-value = {p_sh:.6f} ({keputusan_normal(p_sh)})")
    
    fig3 = sm.qqplot(resid_stud, line="45")
    plt.title("Q-Q Plot")
    st.pyplot(fig3)

    # --- 4) TABEL ANOVA ---
    st.header("3. Hasil ANOVA (RCBD)")
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    st.table(anova_tbl)

    # --- 5) POST-HOC ---
    st.header("4. Post-Hoc Test (Holm)")
    names = model.params.index.tolist()
    
    def ttest_calc(weights):
        w = np.zeros(len(names))
        for nm, val in weights.items():
            w[names.index(nm)] = val
        res = model.t_test(w)
        return float(res.effect), float(res.tvalue), float(res.pvalue)

    tests = [
        ("Pecah - Medium",   {"C(kualitas)[T.Pecah]": 1.0}),
        ("Premium - Medium", {"C(kualitas)[T.Premium]": 1.0}),
        ("Premium - Pecah",  {"C(kualitas)[T.Premium]": 1.0, "C(kualitas)[T.Pecah]": -1.0}),
    ]

    rows, pvals = [], []
    for label, wd in tests:
        eff, t, p = ttest_calc(wd)
        rows.append([label, eff, t, p])
        pvals.append(p)

    rej, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    
    post_hoc_df = pd.DataFrame(rows, columns=["Pasangan", "Diff", "T-Stat", "P-Value"])
    post_hoc_df["P-Adj (Holm)"] = p_adj
    post_hoc_df["Signifikan"] = rej
    st.dataframe(post_hoc_df)

else:
    st.warning("Silakan unggah file CSV di sidebar untuk memulai.")
