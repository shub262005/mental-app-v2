from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import traceback

app = Flask(__name__)

# ── Load model ONCE at startup ─────────────────────────────────────────────────
MODEL = None
MODEL_COLUMNS = None
SCORE_MIN = 4.0
SCORE_MAX = 9.0
LOAD_ERROR = None

try:
    _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    with open(_model_path, 'rb') as _f:
        _saved = pickle.load(_f)
        MODEL = _saved['model']
        MODEL_COLUMNS = list(_saved['columns'])
        SCORE_MIN = _saved.get('score_min', 4.0)
        SCORE_MAX = _saved.get('score_max', 9.0)
except Exception as _e:
    LOAD_ERROR = f"Model load failed: {_e}\n{traceback.format_exc()}"
# ──────────────────────────────────────────────────────────────────────────────

def score_to_label(score):
    """Convert a raw score (4–9) to a percentage and label."""
    pct = round((score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100, 1)
    pct = max(0, min(100, pct))  # clamp to 0–100
    if pct >= 75:
        label, color = "Excellent", "#10b981"
    elif pct >= 55:
        label, color = "Good", "#84cc16"
    elif pct >= 35:
        label, color = "Fair", "#f59e0b"
    else:
        label, color = "Poor", "#ef4444"
    return pct, label, color

@app.route('/', methods=['GET', 'POST'])
def index():
    if LOAD_ERROR:
        return f"<pre style='color:red;padding:2rem'>{LOAD_ERROR}</pre>", 500

    result = None
    if request.method == 'POST':
        try:
            age      = float(request.form['age'])
            platform = request.form['platform']
            usage    = float(request.form['usage'])
            sleep    = float(request.form['sleep'])

            # Build feature vector manually (no pandas needed at runtime)
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

            raw_score = float(MODEL.predict([input_vec])[0])
            # Clamp to valid training range
            raw_score = max(SCORE_MIN, min(SCORE_MAX, raw_score))
            pct, label, color = score_to_label(raw_score)

            result = {
                'score': round(raw_score, 2),
                'pct': pct,
                'label': label,
                'color': color,
            }

        except Exception as e:
            result = {'error': f"Prediction error: {e}"}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
