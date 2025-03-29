import sys
import spacy
import csv
import numpy as np
import lightgbm as lgb
import joblib

# Load spaCy model (if you don't need the full model, consider using a smaller one)
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Load training data from CSV
training_data = []
with open("train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        training_data.append((row["word"], float(row["cost"])))

# Prepare feature vectors and target values
X = []
y = []
for word, cost in training_data:
    doc = nlp(word)
    X.append(doc.vector)
    y.append(cost)
X = np.array(X)
y = np.array(y)

# Train a simpler LightGBM regressor for faster inference.
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'num_leaves': 15,      # Lower number of leaves for simpler trees.
    'learning_rate': 0.1,   # Slightly faster learning rate.
    'n_estimators': 50,     # Fewer trees.
    'random_state': 42
}
model = lgb.LGBMRegressor(**params)
model.fit(X, y)

# Save the model
joblib.dump(model, "cost_predictor_lgbm_fast.joblib")

def predict_cost_custom(system_word):
    doc = nlp(system_word)
    vec = doc.vector.reshape(1, -1)
    predicted_cost = model.predict(vec)[0]
    return max(predicted_cost, 0)

# Combine all command-line arguments into one test word
if len(sys.argv) > 1:
    test_word = " ".join(sys.argv[1:])
else:
    test_word = "Dust"

print(f"Predicted cost for '{test_word}': {predict_cost_custom(test_word):.2f}")
