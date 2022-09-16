from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('regressorcar.pkl')
onehot = joblib.load('OneHotecar.joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	c = ["Car_Name","Fuel_Type","Seller_Type","Transmission","Owner","Kms_Driven","no_year"]
	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df.iloc[:,0:4])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,4:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	print(int_features)

	return render_template("main.html",prediction_text="Your Vehicle price  is : {}".format(result))


if __name__ == "__main__":
	app.debug=True
	app.run(host = '127.0.0.1', port =7000)