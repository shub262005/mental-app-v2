import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load the dataset from CSV
# Assumes 'social_media_addiction.csv' is in the same directory as this script
df = pd.read_csv('social_media_addiction.csv')

# 2. Select the specific features and target for our assignment
features = ['Age', 'Most_Used_Platform', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
target = 'Mental_Health_Score'

# Drop missing values just to be safe before training
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 3. Handle categorical data simply using pandas get_dummies
# This expands 'Most_Used_Platform' into multiple binary columns (One-Hot Encoding)
X_encoded = pd.get_dummies(X, columns=['Most_Used_Platform'])

# 4. Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_encoded, y)

# 5. Save both the model and the exact feature columns we trained on
# We need the columns to properly reconstruct the features in app.py
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'columns': list(X_encoded.columns)  # plain list — no pandas needed to unpickle
    }, f)

print("Training finished! Model and dummy columns saved to model.pkl successfully.")
