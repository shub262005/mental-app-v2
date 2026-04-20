from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Safely try to load the model
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                model = saved_data['model']
                model_columns = saved_data['columns']
        except FileNotFoundError:
            return "Error: model.pkl not found. Please run train_model.py first."

        # Retrieve input data from the HTML form
        age = float(request.form['age'])
        platform = request.form['platform']
        usage = float(request.form['usage'])
        sleep = float(request.form['sleep'])
        
        # Structure the input exactly like our original features DataFrame
        input_data = pd.DataFrame(
            [[age, platform, usage, sleep]], 
            columns=['Age', 'Most_Used_Platform', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
        )
        
        # Convert categorical data directly via get_dummies
        input_encoded = pd.get_dummies(input_data, columns=['Most_Used_Platform'])
        
        # Realign the input's dummy columns to match the actual training set's columns.
        # This handles categories that might be missing in a single prediction.
        # Unknown platforms/features end up ignored, missing platforms are filled with False/0.
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Make the numerical prediction and extract the resulting value
        pred_value = model.predict(input_encoded)[0]
        prediction = round(pred_value, 2)
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Start the server locally
    app.run(debug=True)
