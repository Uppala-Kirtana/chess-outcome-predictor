"""
train.py — Train a chess outcome predictor.

Loads game data, engineers a few features, trains a classifier,
and saves the trained model + encoders to disk for the API to use later.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Load the raw data ---
df = pd.read_csv("games.csv")

# --- 2. Feature engineering ---
# Rating difference is usually the single strongest predictor in chess —
# capturing it as one number is often more useful than two raw ratings.
df["rating_diff"] = df["white_rating"] - df["black_rating"]

# increment_code looks like "15+2" (15 min base time, 2 sec increment per move).
# Split it into two numeric columns so the model can use them separately.
increment_split = df["increment_code"].str.split("+", expand=True)
df["base_time"] = pd.to_numeric(increment_split[0], errors="coerce")
df["increment"] = pd.to_numeric(increment_split[1], errors="coerce")

# Openings: there are hundreds of unique opening names, many rare.
# Keep the top 20 most common, bucket everything else as "Other" —
# this avoids an overly sparse encoding.
top_openings = df["opening_name"].value_counts().nlargest(20).index
df["opening_grouped"] = df["opening_name"].where(
    df["opening_name"].isin(top_openings), "Other"
)

# Encode the opening name as numbers (models need numeric input, not text).
opening_encoder = LabelEncoder()
df["opening_encoded"] = opening_encoder.fit_transform(df["opening_grouped"])

# Drop rows with missing values in the columns we actually use.
feature_cols = ["rating_diff", "base_time", "increment", "opening_encoded", "turns"]
df = df.dropna(subset=feature_cols + ["winner"])

# --- 3. Prepare features (X) and target (y) ---
X = df[feature_cols]
y = df["winner"]  # values: "white", "black", "draw"

# --- 4. Split into train and test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Train the model ---
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- 6. Evaluate ---
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions))

# --- 7. Save the model and the opening encoder together ---
# We need the encoder later in the API to convert an opening NAME
# (like "Sicilian Defense") into the same numeric code the model expects.
joblib.dump(
    {
        "model": model,
        "opening_encoder": opening_encoder,
        "top_openings": list(top_openings),
        "feature_cols": feature_cols,
    },
    "model.joblib",
)

print("Saved trained model to model.joblib")
