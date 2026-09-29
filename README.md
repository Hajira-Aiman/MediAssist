# 🏥 MediAssist AI: Safety-Aware Healthcare Triage Framework

![MediAssist Banner](https://img.shields.io/badge/Healthcare-Safety_First-blue) ![Python](https://img.shields.io/badge/Python-3.12-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)

**MediAssist AI** is a modular, research-grade healthcare assistant that prioritizes patient safety through a rigorous hybrid architecture. It explicitly rejects the "pure probabilistic ML" approach—which often masks life-threatening conditions behind statistical overlap—in favor of deterministic emergency overrides, classical machine learning, and medical consistency validation.

Our thesis: **In healthcare AI, safety and responsible escalation must always supersede raw prediction accuracy.**

---

## 🛑 The Problem: Why Pure ML is Insufficient
In autonomous medical triage, pure machine learning models (whether classical algorithms or Large Language Models) treat critical symptoms identically to benign ones: as statistical features. 
If a user inputs *"coughing up blood and difficulty breathing"*, an ML model trained on common datasets might predict "Asthma" with 85% confidence due to feature overlap, entirely missing the reality of a severe respiratory emergency. Relying solely on statistical likelihoods for triage leads to a high **Unsafe Prediction Rate (UPR)**.

## 🛡️ The Solution: Defense-in-Depth Architecture
MediAssist AI mitigates these risks using a multi-layered, hybrid safety architecture:

1. **Pre-ML Triage Pipeline (Emergency Rule Engine):**
   - Intercepts raw user input *before* it reaches the ML models.
   - Uses strict deterministic matching against critical medical patterns (e.g., Stroke, Hemorrhagic, Cardiac, Respiratory emergencies).
   - Immediately bypasses the ML pipeline and triggers a **🚨 Critical Medical Alert** UI if danger is detected.

2. **Uncertainty-Aware Classical ML:**
   - For non-emergencies, symptom text is parsed using TF-IDF and classified using a highly calibrated Logistic Regression model (with extensible support for Random Forest, XGBoost, etc.).
   - Returns **Top-K Predictions** rather than a single confident guess, explicitly visualizing uncertainty for the user.

3. **Medical Consistency Validation (Post-ML Safety):**
   - Extracts explicit symptom categories (e.g., "respiratory", "gastrointestinal") from the input.
   - Cross-references the ML-predicted disease category against the detected symptoms.
   - **Implausible Outputs** (e.g., predicting an eye infection when the patient reports fever and a cough) receive a harsh confidence penalty (-90% multiplier) and are suppressed in the final ranking.

---

## 🌟 Key Features

- **Dynamic Glassmorphism UI:** A premium, modern Streamlit interface designed to instill calm while clearly delineating between routine care and emergency escalation.
- **Explainability Logging:** Every emergency trigger, suppressed prediction, and consistency penalty is recorded in structured JSON logs (`logs/safety/`) for complete clinical auditability.
- **Severity-Aware Home Care:** Home remedies and rest recommendations are exclusively provided for low-severity conditions.
- **Conversational Memory:** Maintains symptom context across a session for refined follow-up predictions.

---

## 📁 Project Structure

```text
MediAssist/
├── data/                  # Disease names, categories, and home remedies JSONs
├── docs/                  # Architecture diagrams and viva/research notes
├── logs/safety/           # Audit logs for emergency overrides and consistency checks
├── models/classical/      # Saved TF-IDF vectorizers and trained classifiers
├── notebooks/             # Jupyter notebooks for EDA, error analysis, and benchmarking
├── reports/               # Auto-generated validation reports, CSV metrics, and figures
├── src/
│   ├── conversational/    # Dialogue state, intent detection, and context management
│   ├── explainability/    # SHAP, LIME, and TF-IDF feature attribution
│   ├── frontend/          # Streamlit app (app.py) and custom CSS styling
│   ├── models/            # ML training, benchmarking, and inference pipelines
│   ├── preprocessing/     # Text cleaning and symptom normalization
│   ├── safety/            # EmergencyRuleEngine and MedicalConsistencyValidator
│   ├── uncertainty/       # Confidence, entropy, and margin estimation
│   └── recommendation/    # Severity-based home care engine
└── tests/                 # Pytest suite enforcing critical safety axioms
```

> Only `data/*.json`, `data/processed/ml/label_mapping.json`, and the deployed
> Logistic Regression artifact are committed. The raw datasets (~860 MB) and the
> remaining model pickles are gitignored — see **Model Benchmarks** below.

---

## 📊 Safety Metrics & Validation
Standard ML metrics (Accuracy, F1-Score) are insufficient to measure healthcare readiness. MediAssist AI evaluates itself against a synthetic 500-case validation suite focusing on specialized safety metrics:

- **Emergency Escalation Accuracy (EEA):** % of critical emergencies successfully intercepted and escalated (Target: >95%).
- **Missed Emergency Rate (MER):** % of critical emergencies that slipped past the triage layer (Target: ~0%).
- **Unsafe Prediction Rate (UPR):** % of times the model predicts an inappropriate condition with high confidence.
- **Medical Consistency Score:** How accurately the model aligns output disease categories with input symptom profiles.

*Validation results and visualizations are automatically generated and stored in `reports/final_validation_report.md` and `reports/figures/`.*

---

## 🤖 Model Benchmarks

Five classical models were trained and benchmarked on the same TF-IDF feature space
(~25k India-specific rows, 500+ disease classes). Full numbers live in
`reports/model_benchmarks/leaderboard.csv`.

| Model | Accuracy | F1-score | Top-3 Acc | Train (s) | Latency (s/sample) | Size |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.844** | **0.845** | **0.958** | 59.6 | 0.0000045 | 6.2 MB |
| Random Forest | 0.807 | 0.807 | 0.941 | 164.4 | 0.000186 | 9.7 GB |
| Naive Bayes | 0.805 | 0.799 | 0.944 | 0.5 | 0.0000047 | 12.4 MB |
| Decision Tree | 0.153 | 0.154 | 0.214 | 52.5 | 0.0000027 | 0.7 MB |
| XGBoost | 0.071 | 0.031 | 0.089 | 107.5 | 0.000247 | 17.7 MB |

**Why Logistic Regression is deployed:** it wins on every axis that matters here —
best F1, best Top-3 accuracy (the metric that counts, since the UI surfaces Top-K
rather than a single guess), ~40x faster inference than Random Forest, and 1500x
smaller. Its well-calibrated probabilities also matter more than raw accuracy,
because the consistency validator and uncertainty badges both operate on
confidence scores.

**On the weak performers, stated plainly:** Decision Tree and XGBoost perform far
below the rest. With 500+ classes and sparse high-dimensional TF-IDF input, a
single tree fragments badly, and the XGBoost run did not converge under the
default shallow-depth configuration — its 0.031 F1 reflects an
under-tuned model, not a fair verdict on gradient boosting. Both are kept in the
benchmark for transparency rather than presented as tuned baselines. An SVM was
also trained but is excluded from the leaderboard: it produced no metrics run and
its 235 MB artifact made it impractical to ship.

### Shipped vs. reproducible

Only the deployed **Logistic Regression** artifact is committed (~6 MB), so the app
runs immediately after cloning. Every other model's `metrics.json` /
`classification_report.json` is committed for inspection, but the pickles are not —
they total ~10 GB, dominated by Random Forest. Retrain and re-benchmark all models
with:

```bash
pip install -r requirements-dev.txt      # xgboost, shap, lime
python -m src.models.classical_ml.benchmark
```

This regenerates the pickles into `models/classical/` and refreshes
`reports/model_benchmarks/leaderboard.csv`.

---

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Hajira-Aiman/MediAssist.git
   cd MediAssist
   ```

2. **Create a Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   The app runs without this — the variables only affect the optional
   LLM and training paths.

4. **Run the Application:**
   ```bash
   streamlit run src/frontend/app.py
   ```
   The deployed model ships with the repo, so no training step is required.

5. **Run the Test Suite:**
   ```bash
   pytest tests/ -v
   ```

> **Note on scikit-learn version:** the committed model was pickled with
> scikit-learn 1.7.1, while `requirements.txt` pins 1.4.1 for reproducibility.
> Loading it emits an `InconsistentVersionWarning`. Predictions are unaffected;
> to silence it, either install `scikit-learn==1.7.1` or re-run the benchmark
> to regenerate the artifact locally.

---

## ⚖️ Medical Disclaimer
**MediAssist AI is a research and triage framework, NOT a clinical diagnostic tool.** It is explicitly designed to provide preliminary insights and escalate critical symptoms. Always consult a qualified healthcare professional for medical advice, diagnoses, or treatment.

---

## 📄 License

Released under the [MIT License](LICENSE).
