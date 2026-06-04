import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health in Tech — Analysis",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health in Tech — Analysis & Prediction")
st.markdown("**Dataset:** OSMI Mental Health in Tech Survey 2014")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('mental_health_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('mental_survey.csv')
    # Apply your same cleaning steps
    df.columns = df.columns.str.strip()
    top_10 = df["Country"].value_counts().head(10).index
    df = df[df["Country"].isin(top_10)]
    df = df.drop(columns=["Timestamp", "state", "comments"], errors='ignore')
    df["self_employed"] = df["self_employed"].fillna("No")
    df["work_interfere"] = df["work_interfere"].fillna("Don't know")
    df["Age"] = df["Age"].apply(lambda x: x if 15 <= x <= 80 else None)
    df = df.dropna(subset=["Age"])

    def clean_gender(val):
        val = str(val).strip().lower()
        male = ['m','male','make','man','cis male','msle','mail','malr','maile']
        female = ['f','female','woman','femail','femake','cis female','cis-female/femme']
        if val in male: return 'male'
        elif val in female: return 'female'
        elif val in ['non-binary','enby','fluid','genderqueer','androgyne',
                     'agender','trans woman','trans-female','neuter']: return 'non-binary'
        else: return 'other'

    df["Gender"] = df["Gender"].apply(clean_gender)
    return df

clf = load_model()
df = load_data()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔍 Key Insights", "🤖 Predict Treatment"])

# ── TAB 1: EDA ────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Treatment Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df["treatment"].value_counts().plot(kind="bar", ax=ax, color=["#7F77DD","#1D9E75"])
        ax.set_xlabel("Treatment"); ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    with col2:
        st.subheader("Treatment by Gender")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="Gender", hue="treatment", data=df, ax=ax)
        ax.set_xlabel("Gender"); ax.set_ylabel("Count")
        st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax, color="#7F77DD")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    st.pyplot(fig)

# ── TAB 2: KEY INSIGHTS ───────────────────────────────────────
with tab2:
    st.subheader("📌 Key Findings from Statistical Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Respondents", len(df))
    col2.metric("Seek Treatment", f"{(df['treatment']=='Yes').mean()*100:.1f}%")
    col3.metric("Countries Analysed", df["Country"].nunique())

    st.markdown("""
    **Statistical Findings:**
    - 🔬 **Family history** is the strongest predictor of treatment-seeking (Chi-Square p < 0.001)
    - 🏢 **Company mental health benefits** significantly increase treatment likelihood
    - 👥 **Females** seek treatment at a higher rate (~71%) vs males (~47%)
    - 📊 Only **19%** of tech companies run a wellness programme
    """)

    st.subheader("Treatment Rate by Country")
    country_rate = (df.groupby("Country")["treatment"]
                    .apply(lambda x: (x == "Yes").mean() * 100)
                    .sort_values(ascending=False)
                    .reset_index())
    country_rate.columns = ["Country", "Treatment Rate (%)"]
    country_rate["Treatment Rate (%)"] = country_rate["Treatment Rate (%)"].round(1)
    st.dataframe(country_rate, use_container_width=True)

# ── TAB 3: PREDICTION ─────────────────────────────────────────
with tab3:
    st.subheader("🤖 Will This Employee Seek Mental Health Treatment?")
    st.markdown("Fill in the details below to get a prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["male", "female", "non-binary", "other"])
        country = st.selectbox("Country", sorted(df["Country"].unique()))
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
        family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])
        work_interfere = st.selectbox("Mental Health Interferes with Work?",
                                       ["Never", "Rarely", "Sometimes", "Often", "Don't know"])

    with col2:
        no_employees = st.selectbox("Company Size", ["1-5", "6-25", "26-100",
                                                       "100-500", "500-1000", "More than 1000"])
        remote_work = st.selectbox("Remote Work?", ["No", "Yes"])
        tech_company = st.selectbox("Tech Company?", ["Yes", "No"])
        benefits = st.selectbox("Mental Health Benefits Provided?", ["Yes", "No", "Don't know"])
        care_options = st.selectbox("Care Options Available?", ["Yes", "No", "Not sure"])
        wellness_program = st.selectbox("Wellness Programme?", ["Yes", "No", "Don't know"])

    with col3:
        seek_help = st.selectbox("Resources to Seek Help?", ["Yes", "No", "Don't know"])
        anonymity = st.selectbox("Anonymity Protected?", ["Yes", "No", "Don't know"])
        leave = st.selectbox("Ease of Taking Mental Health Leave?",
                              ["Very easy", "Somewhat easy", "Don't know",
                               "Somewhat difficult", "Very difficult"])
        mental_health_consequence = st.selectbox("Fear of Mental Health Consequence?",
                                                  ["No", "Maybe", "Yes"])
        coworkers = st.selectbox("Discuss with Coworkers?", ["Yes", "No", "Some of them"])
        supervisor = st.selectbox("Discuss with Supervisor?", ["Yes", "No", "Some of them"])

    if st.button("🔮 Predict", type="primary"):
        input_data = pd.DataFrame([{
            "Age": age, "Gender": gender, "Country": country,
            "self_employed": self_employed, "family_history": family_history,
            "work_interfere": work_interfere, "no_employees": no_employees,
            "remote_work": remote_work, "tech_company": tech_company,
            "benefits": benefits, "care_options": care_options,
            "wellness_program": wellness_program, "seek_help": seek_help,
            "anonymity": anonymity, "leave": leave,
            "mental_health_consequence": mental_health_consequence,
            "phys_health_consequence": "No", "coworkers": coworkers,
            "supervisor": supervisor, "mental_health_interview": "No",
            "phys_health_interview": "No", "mental_vs_physical": "Don't know",
            "obs_consequence": "No"
        }])

        prediction = clf.predict(input_data)[0]
        probability = clf.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ **Likely to seek treatment** — Probability: {probability*100:.1f}%")
        else:
            st.warning(f"⚠️ **Unlikely to seek treatment** — Probability: {probability*100:.1f}%")

        st.info("*This is a data-based prediction, not medical advice.*")