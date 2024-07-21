import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression


url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_name =['preg','plas','press','skin','test','mass','pedi','age','class']
df = pd.read_csv(url , names=col_name)
print(df)
X= df.iloc[:,:-1]
# print(x)
Y=df.iloc[:,-1]
# print(y)
X_train,x_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25 ,random_state=101)

model =LogisticRegression()
model.fit(X_train,Y_train)
# result = model.score(X_train,Y_train)
# print(result)


joblib.dump(model, '../webpage/logistic_model.pkl')