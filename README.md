# AI-Driven Cyber Threat Awareness and Detection Prototype
**Lab Assignment 01 (Unit 1) - ENSP355 AI in Cyber Security Lab**
Akshita Goel | Roll No. 2301730231 | B.Tech CSE (AI & ML), Section D | K.R. Mangalam University

## Structure
- `data/cyber_events.csv` - synthetic dataset (5000 events, seed 231)
- `src/setup_check.py` - Task 1: validates the environment
- `src/generate_dataset.py` - creates the dataset
- `src/analysis.py` - Tasks 2-5: EDA, ML, MLP, ethics experiments (saves `figures/` and `results.json`)
- `src/build_report.py` - builds the PDF report
- `requirements.txt`

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/setup_check.py
```

## Run
```bash
python src/generate_dataset.py
python src/analysis.py
python src/build_report.py
```

## Models
Logistic Regression, Decision Tree (depth 5), MLPClassifier (32-16). Metrics: accuracy, precision, recall, F1, training time.
No real user data is used.
