# Fake News Detection System Using Machine Learning with a Web Interface

BCSP-165 Project, Bachelor of Computer Applications, IGNOU

A Flask web application that classifies a news headline or article as **FAKE** or **REAL** and shows a confidence percentage. Text is cleaned and converted to TF-IDF features. Three classifiers (Logistic Regression, Multinomial Naive Bayes, Random Forest) are trained and compared, and the one with the best F1-score is selected automatically.

## Features

- User registration, login and logout (hashed passwords, CSRF protection)
- News check with FAKE/REAL label and confidence score
- Personal history of past checks and feedback on predictions
- Admin panel: dataset upload (CSV), model retraining, model-version metrics and usage statistics

## Tech stack

Python 3.10+, Flask, scikit-learn, Pandas, joblib, SQLite, HTML/CSS/JavaScript

## Project structure

```
fake-news-detection-flask/
├── app.py            routes: auth, detection, history, feedback, admin
├── ml.py             preprocessing, training, model comparison, prediction
├── db.py             SQLite schema (users, news_checks, predictions, model_versions, feedback)
├── requirements.txt
├── templates/        HTML pages (base, index, login, register, history, admin)
├── static/           style.css
├── data/             sample.csv (tiny demo dataset)
└── docs/diagrams/    DFDs, ER diagram, flow chart and UI screens
```

## Setup

```
python -m venv venv
venv\Scripts\activate          (Windows)
source venv/bin/activate       (Linux / Mac)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## First run

1. Log in as the default admin: `admin / admin123`. **Change this password before real use or submission.**
2. Open **Admin** and click **Train model** (uses `data/sample.csv`, a tiny demo set).
3. For real accuracy, download the Kaggle "Fake and Real News Dataset" (`Fake.csv` and `True.csv`), merge them into one CSV with columns `text,label` (label = `FAKE` or `REAL`) and upload it on the Admin page.
4. Register a normal user, paste news text on **Check News** and see results in **My History**.

The trained model (`model.joblib`) and the database (`fakenews.db`) are created automatically on first run and are not stored in this repository.

## Screenshots

| Login | Check news |
| --- | --- |
| ![Login](docs/diagrams/ui_login.png) | ![Check news](docs/diagrams/ui_check.png) |

| Result | Admin panel |
| --- | --- |
| ![Result](docs/diagrams/ui_result.png) | ![Admin](docs/diagrams/ui_admin.png) |

## Analysis and design diagrams

- DFD Level 0: `docs/diagrams/dfd_level0.png`
- DFD Level 1: `docs/diagrams/dfd_level1.png`
- DFD Level 2 (processes 1 to 6): `docs/diagrams/dfd_level2_process1.png` to `dfd_level2_process6.png`
- ER diagram: `docs/diagrams/er_diagram.png`
- Prediction flow chart: `docs/diagrams/flowchart_prediction.png`

![ER Diagram](docs/diagrams/er_diagram.png)

## Note on accuracy

The bundled `data/sample.csv` has only 30 rows, so its accuracy is not meaningful. Report metrics from a real dataset. The prediction is a statistical estimate and a screening aid, not a final verdict on whether news is true.
