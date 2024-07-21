import joblib

model = joblib.load('../webpage/logistic_model.pkl')

re = model.predict([[]])