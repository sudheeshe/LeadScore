from flask import Flask, request, render_template
import pandas as pd
import pickle as pkl
import numpy as np



app = Flask(__name__)


@app.route('/', methods=['GET', "POST"])
def home():
    return render_template('home_page.html')

@app.route('/submit', methods=["POST"])
def result():

    #Taking variables from GUI

    age = int(request.form.get('age'))
    job = str(request.form.get('job'))
    marital = str(request.form.get('marital'))
    education = str(request.form.get('education'))
    housing = str(request.form.get('housing'))
    loan = str(request.form.get('loan'))
    contact = str(request.form.get('contact'))
    month = str(request.form.get('month'))
    day_of_week = str(request.form.get('day_of_week'))
    campaign = str(request.form.get('campaign'))
    previous = str(request.form.get('previous'))
    poutcome = str(request.form.get('poutcome'))
    emp_var_rate = float(request.form.get('emp_var_rate'))
    cons_price_idx = float(request.form.get('cons_price_idx'))
    cons_conf_idx = float(request.form.get('cons_conf_idx'))
    euribor3m = float(request.form.get('euribor3m'))
    nr_employed = float(request.form.get('nr_employed'))

    # Loading pickle files
    cat_encoder = pkl.load(open('D:/Ineuron/Project_workshop/LeadScore/Pickle/categorical_encoder.pkl', 'rb'))
    model = pkl.load(open('D:/Ineuron/Project_workshop/LeadScore/Models/Logistic_Regressor.pkl', 'rb'))




    dataframe = pd.DataFrame(
        [[age, job, marital, education, housing, loan, contact, month, day_of_week, campaign, previous,
          poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]],
        columns= ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'campaign',
                  'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'])

    df_array = cat_encoder.transform(dataframe)

    result_1 = model.predict(df_array)
    result_2 = model.predict_proba(df_array)

    output = ""

    if result_2[0][1]>0.70:
        output = f"It's a HOT LEAD,   There is chances of {np.round(result_2[0][1], 2) * 100}% for conversion"

    elif 0.30 <= result_2[0][1] <= 0.70:
        output = f"It's a  WARM LEAD, There is chances of {np.round(result_2[0][1], 2) * 100}% for conversion"

    elif  result_2[0][1] < 0.30:
        output = f"It's a COLD LEAD: There is chances of only {np.round(result_2[0][1], 2) * 100}% for conversion"

    return render_template('result.html', result=output)






if __name__ == "__main__":
        app.run(debug=True)