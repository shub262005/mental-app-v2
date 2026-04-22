import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# 1. Load dataset
df = pd.read_csv('social_media_addiction.csv')

# 2. Features and target
features = ['Age', 'Most_Used_Platform', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
target = 'Mental_Health_Score'
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 3. One-hot encode the platform column
X_encoded = pd.get_dummies(X, columns=['Most_Used_Platform'])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Save model and feature columns
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'columns': list(X_encoded.columns),
        'score_min': 4.0,
        'score_max': 9.0,
    }, f)

print("Training finished! Model saved to model.pkl.")
