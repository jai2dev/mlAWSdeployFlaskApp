from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_loan():
    'Property_Area','Education', 'Dependents','Credit_History','ApplicantIncome','LoanAmount','Loan_Amount_Term'
    property_Area = float(request.form.get('Property_Area'))
    education = int(request.form.get('Education'))
    dependents = int(request.form.get('Dependents'))
    credit_History = float(request.form.get('Credit_History'))
    applicantIncome = int(request.form.get('ApplicantIncome'))
    loanAmount = int(request.form.get('LoanAmount'))
    loan_Amount_Term = int(request.form.get('Loan_Amount_Term'))

    # prediction
    result = model.predict(np.array([property_Area,education, dependents,credit_History,applicantIncome,loanAmount,loan_Amount_Term]).reshape(1,7))

    if result[0] == 1:
        result = 'You are Eligible'
    else:
        result = 'You are Not Eligible'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
