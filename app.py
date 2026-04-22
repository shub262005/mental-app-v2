from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ── Load model ONCE at startup, not on every request ──────────────────────────
_model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(_model_path, 'rb') as _f:
    _saved = pickle.load(_f)
    MODEL = _saved['model']
    MODEL_COLUMNS = list(_saved['columns'])  # e.g. ['Age', 'Avg_Daily_Usage_Hours', ..., 'Most_Used_Platform_Instagram', ...]
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            age      = float(request.form['age'])
            platform = request.form['platform']
            usage    = float(request.form['usage'])
            sleep    = float(request.form['sleep'])

            # Build feature vector manually — replaces pandas.get_dummies()
            # This removes the pandas runtime dependency (saves ~50 MB on Vercel)
            input_vec = np.zeros(len(MODEL_COLUMNS), dtype=float)

            # Fill numeric features
            for col, val in [
                ('Age', age),
                ('Avg_Daily_Usage_Hours', usage),
                ('Sleep_Hours_Per_Night', sleep),
            ]:
                if col in MODEL_COLUMNS:
                    input_vec[MODEL_COLUMNS.index(col)] = val

            # Fill one-hot encoded platform column
            platform_col = f'Most_Used_Platform_{platform}'
            if platform_col in MODEL_COLUMNS:
                input_vec[MODEL_COLUMNS.index(platform_col)] = 1

            pred_value = MODEL.predict([input_vec])[0]
            prediction = round(float(pred_value), 2)

        except Exception as e:
            import traceback
            prediction = f"ERROR: {str(e)} | {traceback.format_exc()}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
