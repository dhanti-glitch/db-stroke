# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, mannwhitneyu
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stroke Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.info-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid #00BFFF;
}
.info-card h4 { color: #00BFFF; margin: 0 0 6px 0; font-size: 15px; }
.info-card p  { color: #cdd6f4; margin: 0; font-size: 13px; line-height: 1.5; }
.info-card .normal { color: #00e676; font-weight: bold; }
.info-card .warn   { color: #FFA500; font-weight: bold; }
.info-card .danger { color: #FF4B4B; font-weight: bold; }

.stat-badge {
    display: inline-block;
    background: #0f3460;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #90caf9;
    margin: 3px 2px;
}
.insight-box {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 10px;
    border: 1px solid #2d4a7a;
}
.insight-box p { color: #b0c4de; font-size: 13px; margin: 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("stroke_dataset_cleaned_final.csv")
    return df

df = load_data()

# Pre-compute
df["stroke_label"] = df["stroke"].apply(lambda x: "Stroke" if x == 1 else "Tidak Stroke")
bins = [0, 18, 35, 50, 65, 100]
labels_age_bins = ['0-18', '19-35', '36-50', '51-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels_age_bins, right=False)
df["health_risk_score"] = df["hypertension"] + df["heart_disease"]

stroke_pct = df['stroke'].mean() * 100
STROKE_COLOR_MAP = {"Stroke": "#FF4B4B", "Tidak Stroke": "#00BFFF"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=80)
st.sidebar.title("🧠 Stroke Risk Dashboard")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "🤖 Prediksi Stroke",
        "📊 Overview",
        "🔍 EDA & Distribusi",
        "⚠️ Faktor Risiko",
        "🧪 A/B Testing & Evaluasi",
        "📋 Kesimpulan"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Data:** {len(df):,} baris")
st.sidebar.markdown(f"**Fitur:** {df.shape[1]} kolom")
st.sidebar.markdown(f"**Prevalensi Stroke:** {stroke_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: PREDIKSI STROKE
# ══════════════════════════════════════════════════════════════════════════════
if menu == "🤖 Prediksi Stroke":
    st.title("🤖 Prediksi Risiko Stroke")
    st.markdown("Masukkan data pasien untuk memprediksi risiko stroke berdasarkan faktor klinis.")
    st.markdown("---")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.markdown("#### 🩺 Data Klinis")
        age = st.slider("Usia (tahun)", 0, 100, 45)
        hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        glucose = st.number_input("Kadar Glukosa (mg/dL)", 50.0, 300.0, 100.0)
        st.caption("💡 Kadar glukosa normal: 70–100 mg/dL (puasa)")

    with col_input2:
        st.markdown("#### 📏 Hitung BMI Otomatis")
        berat = st.number_input("Berat Badan (kg)", 20.0, 200.0, 65.0)
        tinggi = st.number_input("Tinggi Badan (cm)", 100.0, 250.0, 165.0)
        bmi = berat / (tinggi / 100) ** 2
        st.metric("BMI Terhitung", f"{bmi:.1f}")
        if bmi < 18.5:
            st.info("🔵 Underweight (< 18.5)")
        elif bmi < 25:
            st.success("🟢 Normal (18.5 – 24.9)")
        elif bmi < 30:
            st.warning("🟡 Overweight (25 – 29.9)")
        else:
            st.error("🔴 Obesitas (≥ 30)")

    st.markdown("---")

    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        risk_score = (
            age * 0.03 + glucose * 0.01 + bmi * 0.01
            + hypertension * 2 + heart_disease * 2
        )
        probability = min(risk_score / 10, 1)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Probabilitas Stroke", f"{probability*100:.1f}%")
        col_r2.metric("BMI", f"{bmi:.1f}")
        col_r3.metric("Glukosa", f"{glucose:.0f} mg/dL")
        col_r4.metric("Usia", f"{age} thn")

        if probability > 0.5:
            st.error(f"⚠️ **Risiko Stroke TINGGI** — Segera konsultasi ke dokter.")
        else:
            st.success(f"✅ **Risiko Stroke RENDAH** — Tetap jaga pola hidup sehat.")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Risiko Stroke (%)"},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#FF4B4B" if probability > 0.5 else "#00C853"},
                "steps": [
                    {"range": [0, 30], "color": "#1b4332"},
                    {"range": [30, 60], "color": "#7b4f12"},
                    {"range": [60, 100], "color": "#6b1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Kontribusi faktor
        st.subheader("📊 Kontribusi Tiap Faktor Risiko")
        factors = pd.DataFrame({
            "Faktor": ["Usia", "Glukosa", "BMI", "Hipertensi", "Penyakit Jantung"],
            "Kontribusi": [age*0.03, glucose*0.01, bmi*0.01, hypertension*2, heart_disease*2]
        }).sort_values("Kontribusi", ascending=True)
        fig_bar = px.bar(factors, x="Kontribusi", y="Faktor", orientation='h',
                         color="Kontribusi", color_continuous_scale="Reds",
                         text=factors["Kontribusi"].round(2))
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 Overview":
    st.title("📊 Overview Dataset Stroke Prediction")
    st.markdown("Dataset: **Stroke Prediction Dataset** (Kaggle - fedesoriano) | Sudah melalui Data Wrangling")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pasien", f"{len(df):,}")
    col2.metric("Kasus Stroke", f"{df['stroke'].sum():,}")
    col3.metric("Rata-rata Usia", f"{df['age'].mean():.0f} tahun")
    col4.metric("Rata-rata BMI", f"{df['bmi'].mean():.1f}")
    st.markdown("---")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("🍩 Distribusi Kasus Stroke")
        stroke_counts = df['stroke_label'].value_counts().reset_index()
        stroke_counts.columns = ['Status', 'Jumlah']
        fig = px.pie(stroke_counts, values='Jumlah', names='Status',
                     color='Status', color_discrete_map=STROKE_COLOR_MAP, hole=0.5)
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("📋 Tentang Dataset Ini")
        n_stroke = int(df['stroke'].sum())
        n_total  = len(df)
        pct      = n_stroke / n_total * 100
        st.markdown(f"""
        <div style="background:#1a2744;border-left:4px solid #00BFFF;border-radius:8px;
                    padding:14px 18px;margin-bottom:10px;">
          <div style="color:#00BFFF;font-weight:bold;font-size:14px;margin-bottom:8px;">
            🏥 Apa isi dataset ini?
          </div>
          <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
            Dataset ini berisi catatan kesehatan <b style="color:white">{n_total:,} pasien</b>
            yang masing-masing diukur 11 indikator kesehatan seperti usia, tekanan darah,
            kadar gula darah, dan gaya hidup.<br><br>
            Dari seluruh pasien, sebanyak <b style="color:#FF4B4B">{n_stroke:,} orang ({pct:.1f}%)</b>
            tercatat pernah mengalami stroke.<br><br>
            Tujuan utama: <b style="color:#00e676">memahami faktor apa saja yang meningkatkan
            risiko stroke</b> dan membangun model prediksi dini.
          </p>
        </div>
        <div style="background:#1a2744;border-left:4px solid #FFA500;border-radius:8px;
                    padding:14px 18px;margin-bottom:10px;">
          <div style="color:#FFA500;font-weight:bold;font-size:14px;margin-bottom:8px;">
            📌 Faktor yang diteliti
          </div>
          <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
            <b style="color:white">Kondisi medis:</b> usia, hipertensi, penyakit jantung, kadar gula darah, BMI<br>
            <b style="color:white">Gaya hidup:</b> status merokok, jenis pekerjaan<br>
            <b style="color:white">Demografi:</b> jenis kelamin, status menikah, tempat tinggal
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Statistik ringkas numerik — visual
    st.subheader("📈 Statistik Ringkas Variabel Numerik")
    num_stats = df[['age','avg_glucose_level','bmi']].describe().T.round(2)
    num_stats = num_stats[['mean','std','min','50%','max']]
    num_stats.columns = ['Rata-rata','Std Dev','Min','Median','Max']
    num_stats.index = ['Usia (tahun)', 'Glukosa (mg/dL)', 'BMI']

    fig_stat = go.Figure()
    colors = ["#00BFFF", "#FFA500", "#FF4B4B"]
    for i, (idx, row) in enumerate(num_stats.iterrows()):
        fig_stat.add_trace(go.Bar(
            name=idx, x=['Min','Rata-rata','Median','Max'],
            y=[row['Min'], row['Rata-rata'], row['Median'], row['Max']],
            marker_color=colors[i], text=[row['Min'],row['Rata-rata'],row['Median'],row['Max']],
            textposition='outside'
        ))
    fig_stat.update_layout(barmode='group', height=360,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_stat, use_container_width=True)

    st.markdown("""
    <div style="background:#1a2744;border-left:4px solid #00BFFF;border-radius:8px;
                padding:12px 16px;margin-bottom:10px;">
      <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
        💡 <b style="color:#00BFFF">Cara membaca grafik ini:</b> Setiap kelompok batang
        menunjukkan nilai statistik untuk tiga variabel utama — Usia, Kadar Gula Darah (Glukosa),
        dan Berat Badan Proporsional (BMI).<br>
        <b style="color:white">Rata-rata pasien</b> berusia ~43 tahun, glukosa ~106 mg/dL, dan BMI ~29.
        Nilai ini akan dibandingkan lebih dalam di halaman <b>Faktor Risiko</b>.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Data Dictionary — diperkaya untuk user awam
    st.subheader("📖 Panduan Variabel")
    st.markdown("Berikut penjelasan setiap variabel beserta nilai normal/referensi klinis:")

    dict_cards = [
        ("🧬", "gender", "Jenis Kelamin", "Kategorikal", "Male / Female",
         "Jenis kelamin biologis pasien.", None),
        ("🎂", "age", "Usia", "Numerik", "0 – 82 tahun",
         "Usia pasien dalam tahun. Risiko stroke meningkat signifikan di atas 50 tahun.",
         "Risiko tinggi: <span class='danger'>≥ 50 tahun</span>"),
        ("💊", "hypertension", "Hipertensi", "Biner (0/1)", "0 = Tidak, 1 = Ya",
         "Tekanan darah tinggi (≥ 140/90 mmHg). Salah satu faktor risiko stroke terbesar.",
         "Normal: <span class='normal'>< 120/80 mmHg</span> | Tinggi: <span class='danger'>≥ 140/90 mmHg</span>"),
        ("❤️", "heart_disease", "Penyakit Jantung", "Biner (0/1)", "0 = Tidak, 1 = Ya",
         "Riwayat penyakit jantung (seperti gagal jantung, aritmia). Meningkatkan risiko gumpalan darah ke otak.",
         None),
        ("💍", "ever_married", "Status Menikah", "Kategorikal", "Yes / No",
         "Status pernikahan pasien. Berkorelasi dengan usia (orang yang pernah menikah cenderung lebih tua).",
         None),
        ("💼", "work_type", "Jenis Pekerjaan", "Kategorikal", "Private, Self-employed, Govt_job, children, Never_worked",
         "Tipe pekerjaan dapat memengaruhi tingkat stres dan gaya hidup, yang berhubungan dengan risiko stroke.",
         None),
        ("🏙️", "Residence_type", "Tipe Tempat Tinggal", "Kategorikal", "Urban / Rural",
         "Tempat tinggal perkotaan atau pedesaan. Dapat memengaruhi akses layanan kesehatan.",
         None),
        ("🍬", "avg_glucose_level", "Kadar Glukosa Rata-rata", "Numerik", "55 – 272 mg/dL",
         "Rata-rata kadar gula darah. Glukosa tinggi (diabetes) meningkatkan risiko stroke.",
         "Normal: <span class='normal'>70–100 mg/dL</span> (puasa) | Pradiabetes: <span class='warn'>100–125</span> | Diabetes: <span class='danger'>≥ 126 mg/dL</span>"),
        ("⚖️", "bmi", "BMI (Body Mass Index)", "Numerik", "10.3 – 97.6",
         "Indeks Massa Tubuh = Berat Badan (kg) ÷ Tinggi Badan² (m²). Mengukur apakah berat badan proporsional.",
         "Kurus: <span class='warn'>< 18.5</span> | Normal: <span class='normal'>18.5–24.9</span> | Overweight: <span class='warn'>25–29.9</span> | Obesitas: <span class='danger'>≥ 30</span>"),
        ("🚬", "smoking_status", "Status Merokok", "Kategorikal", "formerly smoked / never smoked / smokes / Unknown",
         "Merokok merusak pembuluh darah dan meningkatkan risiko stroke iskemik.",
         None),
        ("🎯", "stroke", "Stroke (Target)", "Biner (0/1)", "0 = Tidak Stroke, 1 = Stroke",
         "Variabel target — apakah pasien pernah mengalami stroke. Ini yang ingin diprediksi oleh model.",
         None),
    ]

    for icon, col_name, label, dtype, values, desc, ref in dict_cards:
        ref_html = f"<br><small>{ref}</small>" if ref else ""
        st.markdown(f"""
        <div class="info-card">
          <h4>{icon} {label} <code style="font-size:11px;color:#90caf9">{col_name}</code>
            &nbsp;<span style="font-size:11px;background:#0f3460;border-radius:10px;
            padding:2px 8px;color:#64b5f6">{dtype}</span>
          </h4>
          <p><b style="color:#90caf9">Nilai:</b> {values}</p>
          <p>{desc}{ref_html}</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA & DISTRIBUSI
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔍 EDA & Distribusi":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Variabel Numerik", "🏷️ Variabel Kategorikal", "🔥 Korelasi"])

    with tab1:
        st.subheader("Distribusi Variabel Numerik")
        filter_num = st.radio("Filter Status:", ["Semua", "Stroke", "Tidak Stroke"],
                               horizontal=True, key="fn")
        df_n = df if filter_num == "Semua" else df[df["stroke_label"] == filter_num]

        num_cols  = ['age', 'avg_glucose_level', 'bmi']
        num_labels = {'age': 'Usia (tahun)', 'avg_glucose_level': 'Glukosa (mg/dL)', 'bmi': 'BMI'}

        for col in num_cols:
            # Gunakan barmode='group' agar batang Stroke selalu terlihat
            df_plot = df_n.copy()
            fig = px.histogram(
                df_plot, x=col,
                color='stroke_label' if filter_num == "Semua" else None,
                barmode='group', nbins=30,
                color_discrete_map=STROKE_COLOR_MAP,
                category_orders={"stroke_label": ["Tidak Stroke", "Stroke"]},
                labels={col: num_labels[col], 'stroke_label': 'Status'},
                title=f"Distribusi {num_labels[col]}",
                text_auto=True
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=340, bargap=0.05,
                               xaxis=dict(dtick=10),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Distribusi Variabel Kategorikal")
        filter_cat = st.radio("Filter Status:", ["Semua", "Stroke", "Tidak Stroke"],
                               horizontal=True, key="fc")
        df_c = df if filter_cat == "Semua" else df[df["stroke_label"] == filter_cat]

        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        cat_labels = {'gender':'Jenis Kelamin','ever_married':'Status Menikah',
                      'work_type':'Jenis Pekerjaan','Residence_type':'Tipe Tempat Tinggal',
                      'smoking_status':'Status Merokok'}

        for col in cat_cols:
            if filter_cat == "Semua":
                grp = df_c.groupby([col,'stroke_label']).size().reset_index(name='count')
                fig = px.bar(grp, x=col, y='count', color='stroke_label', barmode='group',
                             color_discrete_map=STROKE_COLOR_MAP,
                             labels={'count':'Jumlah', col:cat_labels[col], 'stroke_label':'Status'},
                             title=f"Distribusi {cat_labels[col]}", text_auto=True)
            else:
                grp = df_c.groupby(col).size().reset_index(name='count')
                clr = "#FF4B4B" if filter_cat == "Stroke" else "#00BFFF"
                fig = px.bar(grp, x=col, y='count',
                             color_discrete_sequence=[clr],
                             labels={'count':'Jumlah', col:cat_labels[col]},
                             title=f"Distribusi {cat_labels[col]} — {filter_cat}", text_auto=True)
            fig.update_traces(textposition='outside')
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Heatmap Korelasi Antar Variabel Numerik")
        corr = df[['age','avg_glucose_level','bmi','hypertension','heart_disease','stroke']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)
        ax.set_facecolor('#0d1117')
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    annot_kws={"color": "white", "size": 11, "weight": "bold"})
        ax.set_title("Heatmap Korelasi", fontsize=14, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)

        st.markdown("**Insight:** `age` memiliki korelasi tertinggi dengan `stroke`, diikuti `hypertension` dan `avg_glucose_level`.")

        # Interpretasi tabel korelasi terhadap stroke
        st.markdown("#### 📊 Interpretasi Korelasi terhadap Stroke")
        st.markdown("Tabel berikut menunjukkan seberapa kuat hubungan tiap variabel dengan stroke, diurutkan dari yang paling berpengaruh:")

        corr_stroke = corr['stroke'].drop('stroke').abs().sort_values(ascending=False)
        label_map = {
            'age': 'Usia',
            'hypertension': 'Hipertensi',
            'avg_glucose_level': 'Kadar Glukosa',
            'heart_disease': 'Penyakit Jantung',
            'bmi': 'BMI'
        }
        def interpret(v):
            if v >= 0.4: return "🔴 Kuat"
            elif v >= 0.2: return "🟠 Sedang"
            elif v >= 0.1: return "🟡 Lemah"
            else: return "⚪ Sangat Lemah"

        rows = []
        for var, val in corr_stroke.items():
            rows.append({
                "Variabel": label_map.get(var, var),
                "Korelasi dengan Stroke": round(val, 3),
                "Kekuatan Hubungan": interpret(val),
                "Artinya": "Makin tinggi nilainya → makin erat hubungannya dengan stroke"
            })
        interp_df = pd.DataFrame(rows)
        st.dataframe(interp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#1a2744;border-left:4px solid #FFA500;border-radius:8px;
                    padding:12px 16px;margin-top:8px;">
          <p style="color:#cdd6f4;font-size:13px;margin:0;line-height:1.7">
            💡 <b style="color:#FFA500">Catatan:</b> Nilai korelasi 0.04 (misalnya BMI vs stroke) bukan berarti BMI tidak penting —
            hanya berarti hubungan <i>langsungnya</i> kecil. BMI memengaruhi stroke secara <b>tidak langsung</b>
            melalui hipertensi dan diabetes. Korelasi di atas 0.25 umumnya sudah dianggap bermakna dalam data kesehatan.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FAKTOR RISIKO
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "⚠️ Faktor Risiko":
    st.title("⚠️ Analisis Faktor Risiko Stroke")
    st.markdown("Setiap faktor di bawah disertai visualisasi distribusi dan penjelasan mengapa faktor ini berpengaruh.")
    st.markdown("---")

    # ── 1. USIA ────────────────────────────────────────────────────────────────
    st.subheader("1. 🎂 Usia — Faktor Risiko Terkuat")
    age_stroke = df.groupby(['age_group','stroke_label']).size().reset_index(name='count')
    fig = px.bar(age_stroke, x='age_group', y='count', color='stroke_label', barmode='group',
                 color_discrete_map=STROKE_COLOR_MAP,
                 labels={'count':'Jumlah','age_group':'Kelompok Usia','stroke_label':'Status'},
                 title="Distribusi Stroke per Kelompok Usia", text_auto=True)
    fig.update_traces(textposition='outside')
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    <p>📌 <b>Mengapa usia berpengaruh?</b> Seiring bertambahnya usia, pembuluh darah menjadi lebih kaku dan
    rentan mengalami penyempitan (aterosklerosis). Pasien stroke dalam dataset ini rata-rata berusia
    <b>~68 tahun</b>, jauh lebih tua dari yang tidak stroke (~41 tahun). Risiko meningkat tajam pada
    kelompok <b>51–65 tahun</b> dan <b>65+ tahun</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. GLUKOSA ────────────────────────────────────────────────────────────
    st.subheader("2. 🍬 Kadar Glukosa — Indikator Diabetes")
    fig = px.histogram(df, x='avg_glucose_level', color='stroke_label', barmode='group',
                       nbins=30, color_discrete_map=STROKE_COLOR_MAP,
                       category_orders={"stroke_label": ["Tidak Stroke", "Stroke"]},
                       labels={'avg_glucose_level':'Kadar Glukosa (mg/dL)','stroke_label':'Status'},
                       title="Distribusi Kadar Glukosa berdasarkan Status Stroke",
                       text_auto=True)
    fig.update_traces(textposition='outside')
    fig.update_layout(height=340, xaxis=dict(dtick=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    <p>📌 <b>Mengapa glukosa berpengaruh?</b> Kadar gula darah tinggi (diabetes) merusak dinding
    pembuluh darah dan membuat darah lebih mudah membeku. Pasien stroke rata-rata memiliki glukosa
    <b>~133 mg/dL</b> vs ~103 mg/dL pada yang tidak stroke. Batas waspada: <b>≥ 126 mg/dL</b> (diabetes).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. HIPERTENSI & JANTUNG ───────────────────────────────────────────────
    st.subheader("3. 💊 Hipertensi & Penyakit Jantung")

    # Feature Engineering: Health Risk Score — diintegrasikan di sini
    st.markdown("**Health Risk Score** = gabungan hipertensi + penyakit jantung (0, 1, atau 2)")
    hrs = df.groupby(['health_risk_score','stroke_label']).size().reset_index(name='count')
    hrs['health_risk_score'] = hrs['health_risk_score'].map({0:'Tidak Ada (0)', 1:'Salah Satu (1)', 2:'Keduanya (2)'})
    fig_fe = px.bar(hrs, x='health_risk_score', y='count', color='stroke_label', barmode='group',
                    color_discrete_map=STROKE_COLOR_MAP,
                    labels={'health_risk_score':'Risk Score','count':'Jumlah','stroke_label':'Status'},
                    title="Health Risk Score (Hipertensi + Penyakit Jantung) vs Stroke",
                    text_auto=True)
    fig_fe.update_traces(textposition='outside')
    fig_fe.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_fe, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        ht = df.groupby(['hypertension','stroke_label']).size().reset_index(name='count')
        ht['hypertension'] = ht['hypertension'].map({0:'Tidak',1:'Ya'})
        fig = px.bar(ht, x='hypertension', y='count', color='stroke_label', barmode='group',
                     color_discrete_map=STROKE_COLOR_MAP, title="Hipertensi vs Stroke",
                     labels={'hypertension':'Hipertensi','count':'Jumlah','stroke_label':'Status'},
                     text_auto=True)
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hd = df.groupby(['heart_disease','stroke_label']).size().reset_index(name='count')
        hd['heart_disease'] = hd['heart_disease'].map({0:'Tidak',1:'Ya'})
        fig = px.bar(hd, x='heart_disease', y='count', color='stroke_label', barmode='group',
                     color_discrete_map=STROKE_COLOR_MAP, title="Penyakit Jantung vs Stroke",
                     labels={'heart_disease':'Penyakit Jantung','count':'Jumlah','stroke_label':'Status'},
                     text_auto=True)
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <p>📌 <b>Mengapa berpengaruh?</b> Hipertensi memberikan tekanan berlebih pada dinding pembuluh darah
    otak, sedangkan penyakit jantung meningkatkan risiko gumpalan darah yang bisa menyumbat arteri otak.
    Proporsi hipertensi pada pasien stroke (26.5%) jauh lebih tinggi vs yang tidak stroke (9.5%).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. BMI ────────────────────────────────────────────────────────────────
    st.subheader("4. ⚖️ BMI — Peran Tidak Langsung")
    fig = px.histogram(df, x='bmi', color='stroke_label', barmode='group',
                       nbins=30, color_discrete_map=STROKE_COLOR_MAP,
                       category_orders={"stroke_label": ["Tidak Stroke", "Stroke"]},
                       labels={'bmi':'BMI','stroke_label':'Status'},
                       title="Distribusi BMI berdasarkan Status Stroke", text_auto=True)
    fig.update_traces(textposition='outside')
    fig.update_layout(height=320, xaxis=dict(dtick=10),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    <p>📌 <b>Mengapa BMI berpengaruh?</b> Kelebihan berat badan (obesitas, BMI ≥ 30) meningkatkan
    risiko hipertensi dan diabetes — keduanya faktor langsung stroke. Namun dalam dataset ini,
    perbedaan BMI antara kelompok stroke dan tidak stroke tidak sebesar usia atau glukosa.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. SCATTER ────────────────────────────────────────────────────────────
    st.subheader("5. 🔵 Scatter: Usia vs Kadar Glukosa")
    fig = px.scatter(df, x='age', y='avg_glucose_level', color='stroke_label',
                     color_discrete_map=STROKE_COLOR_MAP, opacity=0.5,
                     labels={'age':'Usia','avg_glucose_level':'Kadar Glukosa','stroke_label':'Status'},
                     title="Kombinasi Usia + Glukosa terhadap Risiko Stroke")
    fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
    <p>📌 Pasien stroke cenderung berusia <b>tua</b> sekaligus memiliki <b>glukosa tinggi</b>.
    Kombinasi keduanya (pojok kanan atas scatter plot) adalah zona paling berisiko.</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: A/B TESTING & EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🧪 A/B Testing & Evaluasi":
    st.title("🧪 A/B Testing & Evaluasi Model")
    st.markdown("""
    **A/B Testing** di sini menjawab: *"Apakah faktor risiko tertentu secara statistik
    berhubungan dengan stroke?"* Hasilnya kemudian divalidasi lewat **Confusion Matrix**
    yang menunjukkan performa prediksi model.
    """)
    st.markdown("---")

    # ── A/B TESTING ───────────────────────────────────────────────────────────
    st.subheader("📊 A/B Testing — Uji Statistik Faktor Risiko")

    ab_var = st.selectbox("Pilih variabel yang ingin diuji:", [
        "hypertension", "heart_disease", "ever_married", "gender"
    ], format_func=lambda x: {
        "hypertension": "Hipertensi",
        "heart_disease": "Penyakit Jantung",
        "ever_married": "Status Menikah",
        "gender": "Jenis Kelamin"
    }[x])

    contingency = pd.crosstab(df[ab_var], df["stroke"])
    chi2, p, dof, expected = chi2_contingency(contingency)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi-Square", f"{chi2:.3f}")
    # Format p-value agar mudah dibaca user awam
    if p < 0.0001:
        p_display = "< 0.0001"
    else:
        p_display = f"{p:.4f}"
    col2.metric("P-Value", p_display)
    col3.metric("Derajat Kebebasan (dof)", dof)

    if p < 0.05:
        st.success(f"✅ Terdapat hubungan **signifikan** antara `{ab_var}` dan stroke (tolak H₀, p < 0.05)")
        if p < 0.0001:
            st.caption("ℹ️ P-value sangat kecil (< 0.0001) artinya hubungan ini hampir pasti bukan kebetulan — bukti statistiknya sangat kuat.")
    else:
        st.warning(f"⚠️ Tidak terdapat hubungan signifikan antara `{ab_var}` dan stroke (gagal tolak H₀)")

    col_ct1, col_ct2 = st.columns(2)
    with col_ct1:
        st.markdown("**Tabel Kontingensi (Observed)**")
        st.dataframe(contingency, use_container_width=True)
    with col_ct2:
        st.markdown("**Tabel Expected (jika tidak ada hubungan)**")
        exp_df = pd.DataFrame(expected.round(1),
                               index=contingency.index, columns=contingency.columns)
        st.dataframe(exp_df, use_container_width=True)

    st.markdown("""
    **Hipotesis:**
    - **H₀**: Tidak ada hubungan antara variabel tersebut dengan stroke
    - **H₁**: Ada hubungan antara variabel tersebut dengan stroke
    > Chi-Square membandingkan distribusi observasi dengan distribusi yang diharapkan jika
    > tidak ada hubungan. Semakin besar chi-square & semakin kecil p-value → hubungan makin kuat.
    """)

    st.markdown("---")

    # ── CONFUSION MATRIX ─────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix — Evaluasi Prediksi Model")
    st.markdown("Model memprediksi risiko stroke setiap pasien. Confusion matrix menunjukkan seberapa akurat prediksi tersebut.")

    # Simulasi prediksi
    def predict_stroke(row):
        score = row['age']*0.03 + row['avg_glucose_level']*0.01 + row['bmi']*0.01 \
                + row['hypertension']*2 + row['heart_disease']*2
        return 1 if min(score/10, 1) > 0.5 else 0

    df['y_pred'] = df.apply(predict_stroke, axis=1)
    df['y_pred_label'] = df['y_pred'].map({1:"Stroke", 0:"Tidak Stroke"})

    TP = int(((df['stroke']==1) & (df['y_pred']==1)).sum())
    TN = int(((df['stroke']==0) & (df['y_pred']==0)).sum())
    FP = int(((df['stroke']==0) & (df['y_pred']==1)).sum())
    FN = int(((df['stroke']==1) & (df['y_pred']==0)).sum())
    total = TP+TN+FP+FN
    accuracy  = (TP+TN)/total
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    # Confusion matrix plot — 4 warna berbeda yang jelas
    z = [[TP, FN], [FP, TN]]
    labels_cm = [
        [f"<b>TRUE POSITIVE</b><br>{TP}<br><i>Stroke → Stroke ✅</i>",
         f"<b>FALSE NEGATIVE</b><br>{FN}<br><i>Stroke → Tidak Stroke ❌</i>"],
        [f"<b>FALSE POSITIVE</b><br>{FP}<br><i>Tidak → Stroke ❌</i>",
         f"<b>TRUE NEGATIVE</b><br>{TN}<br><i>Tidak → Tidak ✅</i>"]
    ]

    fig_cm = go.Figure()
    cell_colors = [["#2563a8", "#a84444"], ["#a86c20", "#2a7a50"]]
    for i in range(2):
        for j in range(2):
            fig_cm.add_shape(type="rect",
                x0=j, x1=j+1, y0=1-i, y1=2-i,
                fillcolor=cell_colors[i][j], line=dict(color="white", width=2))
            fig_cm.add_annotation(
                x=j+0.5, y=1.5-i, text=labels_cm[i][j],
                showarrow=False, font=dict(color="white", size=13), align="center")

    fig_cm.update_layout(
        title="Confusion Matrix — Prediksi vs Aktual",
        xaxis=dict(tickvals=[0.5,1.5], ticktext=["Prediksi: Stroke","Prediksi: Tidak Stroke"],
                   showgrid=False, zeroline=False, color="white"),
        yaxis=dict(tickvals=[0.5,1.5], ticktext=["Aktual: Tidak Stroke","Aktual: Stroke"],
                   showgrid=False, zeroline=False, color="white"),
        height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Legenda warna
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.markdown("🟦 **True Positive** — Benar prediksi Stroke")
    col_l2.markdown("🟥 **False Negative** — Stroke tidak terdeteksi *(berbahaya!)*")
    col_l3.markdown("🟧 **False Positive** — False alarm Stroke")
    col_l4.markdown("🟩 **True Negative** — Benar prediksi Tidak Stroke")

    st.markdown("---")

    # Metrik evaluasi
    st.subheader("📐 Metrik Evaluasi Model")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Akurasi", f"{accuracy*100:.1f}%")
    mc2.metric("Precision", f"{precision*100:.1f}%")
    mc3.metric("Recall (Sensitivity)", f"{recall*100:.1f}%",
               help="Seberapa banyak kasus stroke yang berhasil terdeteksi")
    mc4.metric("F1-Score", f"{f1*100:.1f}%")

    st.info("""
    💡 **Catatan:** Untuk deteksi penyakit berbahaya seperti stroke,
    **Recall** lebih penting dari Accuracy. Lebih baik ada *false alarm* (FP tinggi)
    daripada **melewatkan kasus stroke yang nyata (FN tinggi)**.
    """)

    # Filter detail prediksi
    st.markdown("---")
    st.subheader("🔍 Detail Prediksi per Pasien")
    filter_pred = st.radio("Tampilkan:", ["Semua", "True Positive (Benar: Stroke)",
                                           "True Negative (Benar: Tidak Stroke)",
                                           "False Negative (Stroke Terlewat)",
                                           "False Positive (False Alarm)"], horizontal=False)

    filter_map = {
        "Semua": df,
        "True Positive (Benar: Stroke)":     df[(df['stroke']==1) & (df['y_pred']==1)],
        "True Negative (Benar: Tidak Stroke)":df[(df['stroke']==0) & (df['y_pred']==0)],
        "False Negative (Stroke Terlewat)":   df[(df['stroke']==1) & (df['y_pred']==0)],
        "False Positive (False Alarm)":       df[(df['stroke']==0) & (df['y_pred']==1)],
    }
    show_df = filter_map[filter_pred][['age','hypertension','heart_disease',
                                        'avg_glucose_level','bmi','stroke_label','y_pred_label']].head(20)
    show_df.columns = ['Usia','Hipertensi','P.Jantung','Glukosa','BMI','Aktual','Prediksi']
    st.dataframe(show_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: KESIMPULAN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📋 Kesimpulan":
    st.title("📋 Kesimpulan & Rekomendasi")
    st.markdown("---")

    insights = [
        ("🔴", "Usia adalah faktor risiko terkuat",
         "Pasien stroke rata-rata berusia ~68 tahun vs ~41 tahun. Risiko meningkat signifikan pada usia 51+."),
        ("🟠", "Kadar glukosa tinggi meningkatkan risiko",
         "Pasien stroke rata-rata glukosa ~133 mg/dL vs ~103 mg/dL. Waspada jika ≥ 126 mg/dL (diabetes)."),
        ("🟡", "Hipertensi berkontribusi signifikan",
         "Proporsi hipertensi pada pasien stroke 26.5% vs 9.5% — terbukti signifikan lewat Chi-Square."),
        ("🟢", "Penyakit jantung memperparah risiko",
         "Pasien dengan riwayat jantung memiliki kemungkinan stroke lebih tinggi."),
        ("🔵", "BMI berperan namun tidak dominan",
         "BMI berpengaruh tidak langsung via risiko hipertensi dan diabetes."),
        ("🟣", "Status merokok & pekerjaan berpengaruh",
         "'Formerly smoked' dan 'Private worker' memiliki proporsi stroke lebih tinggi."),
    ]

    for icon, title, desc in insights:
        with st.expander(f"{icon} {title}", expanded=True):
            st.write(desc)

    st.markdown("---")
    st.subheader("📌 Rekomendasi")
    st.markdown("""
    1. **Skrining rutin** untuk populasi usia >50 tahun dengan hipertensi atau penyakit jantung.
    2. **Pengelolaan kadar glukosa** khususnya pada penderita diabetes (glukosa ≥ 126 mg/dL).
    3. **Gaya hidup sehat** — berhenti merokok, menjaga BMI 18.5–24.9.
    4. **Fitur utama model prediksi:** `age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`.
    5. **Prioritaskan Recall** dalam evaluasi model stroke — jangan sampai kasus positif terlewat.
    """)

    st.markdown("---")
    st.caption("Dashboard dibuat untuk keperluan Data Science | Stroke Prediction Dataset (Kaggle - fedesoriano)")
