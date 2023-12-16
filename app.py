from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_loan():
    'Property_Area','Education', 'Dependents','Credit_History','ApplicantIncome','LoanAmount','Loan_Amount_Term'
    property_Area = request.form.get('Property_Area')
    education = request.form.get('Education')
    dependents = request.form.get('Dependents')
    credit_History = float(request.form.get('Credit_History'))
    applicantIncome = int(request.form.get('ApplicantIncome'))
    loanAmount = float(request.form.get('LoanAmount'))
    loan_Amount_Term = float(request.form.get('Loan_Amount_Term'))
    inputData=np.array([property_Area,education, dependents,credit_History,applicantIncome,loanAmount,loan_Amount_Term])
    categorical_data = inputData[:3]
    numerical_data = inputData[3:]
    label_encoder = LabelEncoder()
    encoded_categorical_data = label_encoder.fit_transform(categorical_data)
    # prediction
    encoded_categorical_data=np.array(encoded_categorical_data,dtype=object)
    final_input_data = np.concatenate((encoded_categorical_data, numerical_data))
    final_input_data=final_input_data.reshape(1,-1)
    result = model.predict(final_input_data)
    if result[0] == 1:
        result = 'You are Eligible'
    else:
        result = 'You are Not Eligible'

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8090)
