import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Simulate dataset
X = np.random.randint(0, 2, size=(500, 20))
symptom_counts = X.sum(axis=1).reshape(-1, 1)
X = np.hstack((X, symptom_counts))

diseases = ['COVID-19', 'Dengue', 'Diabetes', 'Flu', 'Hypertension', 'Malaria']
y = np.random.choice(diseases, size=500)

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Evaluation
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=le.classes_))

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_preds, target_names=le.classes_))

# Optional test prediction (simulate user input)
# Example: input = 10 symptoms checked => first 10 = 1, rest = 0
sample_input = [1]*10 + [0]*10
symptom_count = sum(sample_input)
sample_input.append(symptom_count)
input_array = np.array(sample_input).reshape(1, -1)
predicted = rf_model.predict(input_array)
print("\nPredicted Disease for sample input:", le.inverse_transform(predicted)[0])
