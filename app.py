with t3:
        st.markdown("<h2 style='text-align: center;'>Analisis Inferensial (RCBD)</h2>", unsafe_allow_html=True)
        
        # Penjelasan Konteks
        st.markdown(f"""
        <div class='stats-card'>
            <h4 style='color:{CHART_PALETTE['moss_green']};'>Konteks Model</h4>
            <p>Penelitian ini menggunakan <b>Rancangan Acak Kelompok (RCBD)</b> untuk menguji apakah terdapat perbedaan harga yang signifikan antar kualitas beras (Premium, Medium, Pecah) dengan memperlakukan <b>Bulan</b> sebagai blok untuk mengontrol variasi musiman.</p>
        </div>
        """, unsafe_allow_html=True)

        # Hitung ANOVA
        long_complete = wide_df.reset_index().melt(id_vars=["Bulan"], var_name="Kualitas", value_name="Harga")
        model = ols('Harga ~ C(Kualitas) + C(Bulan)', data=long_complete).fit()
        aov_table = anova_lm(model, typ=2)
        
        # Penyesuaian Nama Index agar sesuai output yang diminta
        aov_display = aov_table.copy()
        aov_display.index = ['C(kualitas)', 'C(bulan)', 'Residual']
        
        # Tampilan Tabel ANOVA yang Menarik dengan format kolom Sum_sq, df, F, PR(>F)
        st.markdown("#### === ANOVA (harga ~ kualitas + bulan) ===")
        
        # Memformat tabel: Sum_sq dan PR(>F) menggunakan notasi ilmiah, sisanya desimal/float
        formatted_table = aov_display.style.format({
            "sum_sq": "{:.6e}",
            "df": "{:.1f}",
            "F": "{:.6f}",
            "PR(>F)": "{:.6e}"
        }).set_properties(**{
            'background-color': 'white',
            'color': CHART_PALETTE['dark_text'],
            'border-color': CHART_PALETTE['warm_grey']
        })
        
        st.table(formatted_table)

        # Tampilan Parameter Uji di Kolom (Opsional tetap dipertahankan untuk visualisasi cepat)
        p_val = aov_table.loc["C(Kualitas)", "PR(>F)"]
        f_stat = aov_table.loc["C(Kualitas)", "F"]
        
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown(f"""
            <div style='background-color:{CHART_PALETTE['moss_green']}; color:white; padding:15px; border-radius:10px; text-align:center;'>
                <small>F-Statistic (Kualitas)</small><br>
                <b style='font-size:20px;'>{f_stat:.6f}</b>
            </div>
            """, unsafe_allow_html=True)
        with col_right:
             st.markdown(f"""
            <div style='background-color:{CHART_PALETTE['moss_green']}; color:white; padding:15px; border-radius:10px; text-align:center;'>
                <small>P-Value (Kualitas)</small><br>
                <b style='font-size:20px;'>{p_val:.6e}</b>
            </div>
            """, unsafe_allow_html=True)

        # Interpretasi Visual
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Interpretasi Hipotesis")
        
        if p_val < 0.05:
            st.markdown(f"""
            <div style='border-left: 10px solid {CHART_PALETTE['moss_green']}; background-color: #e8f5e9; padding: 20px;'>
                <h4 style='color:#2e7d32; margin:0;'>H<sub>1</sub> Diterima (Signifikan)</h4>
                <p style='margin:10px 0 0 0;'>Terdapat bukti statistik yang sangat kuat bahwa <b>setidaknya satu kategori kualitas memiliki harga rata-rata yang berbeda secara nyata</b> dibandingkan kategori lainnya pada tingkat kepercayaan 95%.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='border-left: 10px solid #c62828; background-color: #ffeae8; padding: 20px;'>
                <h4 style='color:#c62828; margin:0;'>H<sub>0</sub> Gagal Ditolak (Tidak Signifikan)</h4>
                <p style='margin:10px 0 0 0;'>Tidak ditemukan perbedaan harga yang signifikan antar kualitas beras.</p>
            </div>
            """, unsafe_allow_html=True)
