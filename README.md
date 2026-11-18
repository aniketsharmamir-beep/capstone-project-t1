# Titanic Survival Analysis: Capstone Project

## Objective
Find which passenger characteristics drive survival on the Titanic, so safety planners can identify the groups at highest risk and prioritize evacuation resources.

## Dataset
Titanic dataset (Kaggle): 891 passengers, 12 columns.

## Installation
1. pip install -r requirements.txt
2. python src/data_preprocessing.py
3. Open notebooks/capstone_analysis.ipynb and run all cells.

## Key Findings
- Only 38% of passengers survived.
- Women survived far more often than men (74% vs 19%).
- Survival fell with ticket class: 63% (1st), 47% (2nd), 24% (3rd).
- Logistic Regression was the best model: accuracy 0.832, F1-score 0.769.

## Files
- src/data_preprocessing.py: cleaning, outlier capping, feature engineering, encoding, scaling
- notebooks/capstone_analysis.ipynb: EDA, modeling, tuning, evaluation
- reports/: final report and presentation slides
- images/: charts
