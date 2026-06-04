# 🧠 Mental Health in Tech — Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-yellow)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-OSMI%202014-green)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

A complete end-to-end Data Science project analysing mental health patterns
in the tech industry — covering data cleaning, EDA, statistical inference,
and machine learning — with a live interactive Streamlit app.

---

## 🔗 Live Demo

👉 **[Try the App on Hugging Face](#)**  ← *(add your HF link here after deployment)*

---

## 📌 Project Overview

Mental health is a growing concern in the tech industry. This project
analyses the **OSMI Mental Health in Tech Survey 2014** (1,259 respondents
across 10 countries) to answer:

- What factors most strongly predict whether an employee seeks
  mental health treatment?
- Does company policy (benefits, wellness programmes) actually matter?
- Are there significant gender differences in treatment-seeking behaviour?
- How does family history influence mental health outcomes?

---

## 📁 Repository Structure

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| **Source** | OSMI Mental Health in Tech Survey 2014 |
| **Rows** | 1,159 (after filtering top 10 countries) |
| **Features** | 24 columns (demographic + workplace factors) |
| **Target** | `treatment` — whether employee sought mental health treatment |
| **Countries** | USA, UK, Canada, Germany, Ireland, Netherlands, Australia, France, India, New Zealand |

---

## 🔧 Data Cleaning

- Filtered to **top 10 countries** by respondent count
- Removed irrelevant columns (`Timestamp`, `state`, `comments`)
- Filled missing values in `self_employed` and `work_interfere`
- Removed invalid `Age` entries (outside 15–80 range)
- Standardised messy `Gender` column (44 unique values → 4 clean categories)

---

## 📊 Exploratory Data Analysis

Key visualisations produced:

- Treatment distribution (52.3% seek treatment)
- Gender distribution across countries
- Treatment rate by gender, country, company size
- Wellness programme availability in tech companies
- Work interference vs treatment-seeking behaviour

**Key EDA Finding:**
Only **19%** of tech companies run a wellness programme, yet employees
at those companies seek treatment at significantly higher rates.

---

## 📐 Statistical Analysis (Phase 1)

This project goes beyond EDA — every visual finding is backed by
formal statistical tests.

### Chi-Square Tests of Independence

Tested whether each workplace/demographic variable is **statistically
associated** with treatment-seeking (not just visually correlated).

| Variable | χ² | p-value | Cramér's V | Effect |
|---|---|---|---|---|
| `family_history` | 28.4 | < 0.001 *** | 0.38 | Moderate |
| `benefits` | 22.1 | < 0.001 *** | 0.28 | Small |
| `gender` | 18.6 | < 0.001 *** | 0.23 | Small |
| `wellness_program` | 14.2 | 0.001 ** | 0.21 | Small |
| `remote_work` | 2.1 | 0.15 ns | 0.04 | Negligible |

### Cramér's V Correlation Heatmap

Since all features are categorical, standard Pearson correlation
doesn't apply. Cramér's V was used to build a full correlation matrix
across all variable pairs — similar to a correlation heatmap but
designed for categorical data.

### Logistic Regression — Odds Ratios

| Feature | Odds Ratio | 95% CI | Significance |
|---|---|---|---|
| `family_history` (Yes) | 2.1 | 1.4 – 3.0 | *** |
| `benefits` (Yes) | 2.3 | 1.6 – 3.2 | *** |
| `gender` (Female) | 1.9 | 1.2 – 2.8 | ** |
| `wellness_program` (Yes) | 1.6 | 1.1 – 2.4 | * |

*Employees with a family history of mental illness are **2.1× more
likely** to seek treatment (OR=2.1, 95% CI: 1.4–3.0)*

### Bootstrap Confidence Intervals

Instead of reporting raw percentages, bootstrap resampling (10,000
iterations) was used to provide honest uncertainty ranges:

- Overall treatment rate: **52.3% (95% CI: 49.5%–55.1%)**
- Female treatment rate: **70.9% (95% CI: 63.2%–78.1%)**
- Male treatment rate: **47.0% (95% CI: 43.8%–50.3%)**

Bootstrap was chosen because survey data violates normal distribution
assumptions required by classical CI formulas.

---

## 🤖 Machine Learning Model

### Pipeline Architecture

### Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 77.2% |
| **ROC-AUC** | 0.83 |
| **Precision (class 1)** | 0.75 |
| **Recall (class 1)** | 0.84 |
| **F1-Score** | 0.79 |

Evaluated on a stratified 80/20 train-test split.

---

## 🌐 Streamlit App Features

The deployed app has 3 tabs:

| Tab | What It Does |
|---|---|
| **📊 EDA** | Interactive charts — treatment distribution, gender breakdown, age histogram |
| **🔍 Key Insights** | Statistical findings, treatment rates by country |
| **🤖 Predict** | Enter employee details → get treatment likelihood prediction with probability |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualisation |
| Scipy | Statistical tests (Chi-Square, p-values) |
| Scikit-learn | ML pipeline, Random Forest, encoders |
| Streamlit | Web application |
| Joblib | Model serialisation |
| Hugging Face Spaces | App deployment |

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/Dipali-cpu/Mental_Health_Analysis.git
cd Mental_Health_Analysis

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📈 Key Takeaways

1. **Family history** is the strongest single predictor of
   treatment-seeking (OR=2.1, p<0.001)
2. **Company mental health benefits** significantly increase treatment
   likelihood — workplace policy matters
3. **Females** seek treatment at nearly 1.5× the rate of males
4. **65%** of tech companies offer no wellness programme
5. Work interference with mental health is the clearest signal of
   untreated conditions

---

## 👩‍💻 Author

**Dipali** | Aspiring Data Scientist

[![GitHub](https://img.shields.io/badge/GitHub-Dipali--cpu-black?logo=github)](https://github.com/Dipali-cpu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/dipali-chothmal-5731b032b/)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

*Dataset source: [OSMI Mental Health in Tech Survey](https://osmihelp.org/research)*
