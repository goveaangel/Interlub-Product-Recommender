# 🔬 Interlub Product Recommender
A Streamlit app for grease recommendation and scenario simulation

---

## 📂 Repository Structure

```bashso
INTERLUB/
│
├── data/
│   ├── competidores.csv
│   ├── competidores_info.csv
│   ├── competidores_preprocesados.csv
│   ├── datos_grasas_Interlub_limpios_v2.csv
│   ├── datos_grasas_Tec.csv
│   ├── datos_grasas_Tec_limpio.csv
│   ├── df_categoricas.csv
│   ├── df_numericas.csv
│
├── images/
│   ├── interlub.png
│   └── interlub2.png
│
├── notebooks/
│   ├── 1RetoInterlub_M2003B_student_final.ipynb
│   ├── 2RetoInterlub_M2003B_student (1).ipynb
│   ├── 3RetoInterlub_M2003B.ipynb
│   └── Recomendador.ipynb
│
├── pages/
│   ├── 1_Parametros_del_cliente.py
│   ├── 2_Recomendador.py
│   └── 3_Regresor.py
│
├── scripts/
│   ├── __init__.py
│   ├── creacion_competidores.py
│   ├── recomendador_palabras.py
│   ├── recomendador.py
│   └── regresor.py
│
├── Inicio.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Project Overview

Interlub Product Recommender is a Streamlit application designed to recommend the most suitable Interlub grease based on a client’s operating conditions and to simulate how technical properties would change under hypothetical scenarios.

The application is organized into three main modules:

### 1️⃣ Client Parameters

Users can enter their lubrication requirements in two ways:
- Structured questionnaire: temperature ranges, load severity, environment, water exposure, and other operational factors.
- Free-text description: ideal when the client explains the need informally instead of using technical terminology.

### 2️⃣ Grease Recommender

Using the collected information, the system generates personalized recommendations through:
- Technical similarity in a normalized feature space
- Semantic similarity using TF-IDF + cosine distance on product descriptions
- A global score (0–100) combining thermal compatibility, load/severity, water resistance, and match to the target profile

Results include ranked recommendations, comparison tables, and radar charts for detailed inspection.

### 3️⃣ Scenario Simulator (Regressor)

A regression model trained on synthetic data allows users to:
- Select any grease (ideally the recommended one)
- Modify one technical variable
- Observe how the remaining properties change according to the model’s learned relationships

This enables practical “what-if” analysis without requiring new laboratory measurements.

---

## 📊 Methodology

The system combines feature engineering, semantic similarity, and regression modeling to generate lubricant recommendations and simulate technical changes. The methodology includes:

### 1️⃣ Data Preparation
- Cleaning and standardizing Interlub product data
- Separating categorical and numerical technical features
- Creating enriched textual fields by combining product descriptions, applications, benefits, and other metadata
- Generating a normalized 0–1 scale for technical comparison across features

### 2️⃣ Technical Similarity Model

Numerical variables are transformed into a normalized feature vector for each grease.
Client inputs (form-based questionnaire) are converted into an ideal target vector using predefined scoring rules.

Similarity is computed using:
- Euclidean distance in the normalized feature space
- A global score (0–100) weighting key operational dimensions:
- Thermal behavior
- Load severity
- Water resistance
- Overall proximity to the ideal profile

This produces a ranked list of the most technically appropriate greases.

### 3️⃣ Text-Based Semantic Recommender

For free-text inputs, the system uses:
- TF-IDF vectorization (unigrams + bigrams)
- Cosine similarity between the user’s description and each product’s combined text field

This enables recommendation even when the user provides no structured parameters.

### 4️⃣ Regression-Based Scenario Simulation

A linear regression model is trained using synthetic data generated from the technical feature distributions.
The model is used to:
- Predict how all technical variables change when one feature is modified
- Provide interactive “what-if” simulations inside Streamlit
- Allow users to explore parameter impacts on any grease (ideally the recommended one)

### 5️⃣ Visualization & Interaction

The app integrates:
- Ranking tables
- Radar charts (ideal vs. product)
- Comparison plots within recommended groups
- Scenario plots showing predicted changes from the regression model

---

## 📈 Results Summary

The technical recommender consistently identifies a **top-performing grease** that best matches the client’s operational profile. This selection is based on normalized similarity metrics and weighted scoring across thermal behavior, load severity, water exposure, and overall proximity to the ideal feature vector. In practice, the highest-ranked product is the one with the smallest distance to the target profile, indicating strong technical compatibility.

The semantic (text-based) recommender also performs reliably, selecting products whose **descriptions and applications align closely** with the user’s free-text input. By combining TF-IDF vectorization with cosine similarity, the system captures intent even when the client does not specify structured parameters, returning greases with conceptually similar functional characteristics.

The scenario simulator, powered by a regression model trained on synthetic data, produces **directionally consistent predictions** when a technical variable is modified. Adjusting a single feature results in coherent shifts across the remaining properties, preserving realistic relationships between variables. This enables meaningful “what-if” analysis, particularly when evaluating the behavior of the recommended grease under alternative conditions.

---

## 🧠 Key Insights

- The technical recommender reliably identifies the grease closest to the ideal operating profile, showing that normalized similarity metrics and weighted scoring provide stable, interpretable results.
- Semantic matching proves effective: text-based inputs consistently retrieve products with descriptions and applications aligned to the user’s intentions, even without structured parameters.
- The regression simulator produces coherent, directionally consistent responses when a variable is modified, indicating that the synthetic training process preserves realistic relationships between technical properties.

---

## 🧩 Technologies Used

- Python — core implementation for data processing, modeling, and algorithms
- Streamlit — user interface for the recommender, text analyzer, and scenario simulator
- Pandas & NumPy — dataset handling, preprocessing, and feature engineering
- scikit-learn — TF-IDF vectorization, cosine similarity, and linear regression modeling
- Plotly — radar charts and interactive visualizations
- Git & GitHub — version control and collaboration


---

## 📘 Reports

---

## 👥 Authors

- **Diego Vértiz Padilla**  
- **José Ángel Govea García**  
- **Daniel Alberto Sánchez Fortiz**  
- **Augusto Ley Rodríguez**  

Tecnológico de Monterrey, School of Engineering and Sciences  
Guadalajara, Jalisco — México  