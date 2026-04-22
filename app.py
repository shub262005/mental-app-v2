from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import traceback

app = Flask(__name__)

# ── Load model ONCE at startup ─────────────────────────────────────────────────
MODEL = None
MODEL_COLUMNS = None
LOAD_ERROR = None

try:
    _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    with open(_model_path, 'rb') as _f:
        _saved = pickle.load(_f)
        MODEL = _saved['model']
        MODEL_COLUMNS = list(_saved['columns'])
except Exception as _e:
    LOAD_ERROR = f"Model load failed: {_e}\n{traceback.format_exc()}"
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    # Surface model load errors clearly
    if LOAD_ERROR:
        return f"<pre style='color:red;padding:2rem'>{LOAD_ERROR}</pre>", 500

    prediction = None
    if request.method == 'POST':
        try:
            age      = float(request.form['age'])
            platform = request.form['platform']
            usage    = float(request.form['usage'])
            sleep    = float(request.form['sleep'])

            # Build feature vector manually (no pandas needed)
            input_vec = np.zeros(len(MODEL_COLUMNS), dtype=float)

            for col, val in [
                ('Age', age),
                ('Avg_Daily_Usage_Hours', usage),
                ('Sleep_Hours_Per_Night', sleep),
            ]:
                if col in MODEL_COLUMNS:
                    input_vec[MODEL_COLUMNS.index(col)] = val

            platform_col = f'Most_Used_Platform_{platform}'
            if platform_col in MODEL_COLUMNS:
                input_vec[MODEL_COLUMNS.index(platform_col)] = 1

            pred_value = MODEL.predict([input_vec])[0]
            prediction = round(float(pred_value), 2)

        except Exception as e:
            prediction = f"Prediction error: {e}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
