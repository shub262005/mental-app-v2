import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
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

# 4. Train/test split to validate quality
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Train RandomForestRegressor — much better at capturing non-linear patterns
#    (e.g., the compounding effect of both high usage AND low sleep)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
print(f"R2 Score:  {r2_score(y_test, preds):.4f}")
print(f"MAE:       {mean_absolute_error(y_test, preds):.4f}")

# 7. Quick sanity check
print("\n--- BAD habits: 9h usage, 4h sleep, Instagram, age 20 ---")
bad = pd.DataFrame([[20, 'Instagram', 9.0, 4.0]], columns=features)
bad_enc = pd.get_dummies(bad, columns=['Most_Used_Platform'])
bad_enc = bad_enc.reindex(columns=X_encoded.columns, fill_value=0)
print(f"Predicted: {model.predict(bad_enc)[0]:.2f}  (range 4–9, lower = worse)")

print("\n--- GOOD habits: 1h usage, 8h sleep, LinkedIn, age 22 ---")
good = pd.DataFrame([[22, 'LinkedIn', 1.0, 8.0]], columns=features)
good_enc = pd.get_dummies(good, columns=['Most_Used_Platform'])
good_enc = good_enc.reindex(columns=X_encoded.columns, fill_value=0)
print(f"Predicted: {model.predict(good_enc)[0]:.2f}")

# 8. Save — columns as plain Python list (no pandas dependency for unpickling)
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'columns': list(X_encoded.columns),  # plain list — no pandas needed to unpickle
        'score_min': 4.0,
        'score_max': 9.0,
    }, f)

print("\nTraining finished! Model saved to model.pkl.")
