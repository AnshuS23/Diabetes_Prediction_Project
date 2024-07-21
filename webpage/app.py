from flask import Flask, render_template ,request
import joblib
model = joblib.load('logistic_model.pkl')

app =Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact_us')
def contact():
    return render_template('contact.html')

@app.route('/events')
def events():
    return render_template('events.html')

# @app.route('/submit', methods =['post'])
# def submit():
#     a =request.form.get('Mobile Number')
#     b=request.form.get('email')
#     c=request.form.get('password')
#     print(a ,b, c)
#     return 'Data Collected'


@app.route('/predict', methods =['post'])
def submit():
    preg=int(request.form.get('preg'))
    plas=int(request.form.get('plas'))
    press=int(request.form.get('press'))
    skin = int(request.form.get('skin'))
    test = int(request.form.get('test'))
    mass = float(request.form.get('mass'))
    pedi = float(request.form.get('pedi'))
    age = int(request.form.get('age'))
    re=model.predict([[preg,plas,press,skin,test,mass,pedi,age]])
    if re[0]==1:
        return 'You are Diabetic! Please take care of your Health'
    else:
        return "You don't have Diabetics. You are safe!"

    # return 'Data Collected'
app.run(debug=True , host='0.0.0.0')