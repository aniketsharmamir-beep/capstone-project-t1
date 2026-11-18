"""
Data preprocessing for the Titanic capstone project.

Run directly to clean the raw data and save data/processed_dataset.csv:
    python src/data_preprocessing.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw_dataset.csv"
PROCESSED_PATH = "data/processed_dataset.csv"
NUMERIC_COLS = ["Age", "Fare", "FarePerPerson", "FamilySize"]


def clean_data(df):
    """Handle missing values and drop columns that are not useful."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())            # median is robust to extremes
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])                              # 77% missing
    return df.drop_duplicates()


def cap_outliers(df, column):
    """Cap values outside 1.5 x IQR so extreme values do not distort the model."""
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    df = df.copy()
    df[column] = df[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def engineer_features(df):
    """Create new features from existing columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    return df.drop(columns=["PassengerId", "Name", "Ticket"])


def encode_features(df):
    """Convert text columns to 0/1 columns (one-hot encoding)."""
    return pd.get_dummies(df, columns=["Sex", "Embarked", "Title"], drop_first=True, dtype=int)


def preprocess(df):
    """Full pipeline: clean -> cap outliers -> engineer features -> encode."""
    df = clean_data(df)
    df = cap_outliers(df, "Fare")
    df = engineer_features(df)
    return encode_features(df)


def scale_numeric(X_train, X_test, columns=NUMERIC_COLS):
    """Standardize numeric columns. The scaler learns from the training data only."""
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    processed = preprocess(raw)
    processed.to_csv(PROCESSED_PATH, index=False)
    print("Raw shape:", raw.shape, "-> Processed shape:", processed.shape)
