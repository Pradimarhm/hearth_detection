from flask import Flask, request, render_template
from api.hybrid_prediction import hybrid_predict
import os

# app = Flask(__name__)
# Mendapatkan path folder 'api' secara dinamis
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')

# Beritahu Flask lokasi folder templates
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    data = {
        'Age': int(request.form['age']),
        'Gender': request.form['gender'],
        'Blood Pressure': float(request.form['blood_pressure']),
        'Cholesterol Level': float(request.form['cholesterol_level']),
        'Sleep Hours': float(request.form['sleep_hours']),
        'Smoking': request.form['smoking'],
        'Family Heart Disease': request.form['family_heart_disease'],
        'Alcohol Consumption': request.form['alcohol_consumption'],
        'Exercise Habits': request.form['exercise_habits'],
        'BMI': float(request.form['bmi']),
        'CRP Level': float(request.form['crp_level']),
        'Diabetes': request.form['diabetes'],
        'High Blood Pressure': request.form['high_blood_pressure'],
        'Low HDL Cholesterol': request.form['low_hdl_cholesterol'],
        'High LDL Cholesterol': request.form['high_ldl_cholesterol'],
        'Stress Level': request.form['stress_level'],
        'Sugar Consumption': request.form['sugar_consumption'],
        'Triglyceride Level': float(request.form['triglyceride_level']),
        'Fasting Blood Sugar': float(request.form['fasting_blood_sugar']),
        'Homocysteine Level': float(request.form['homocysteine_level'])
    }

    # Panggil fungsi hybrid_predict
    result = hybrid_predict(data)

    # Kirim hasil ke template
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
    

app = app
