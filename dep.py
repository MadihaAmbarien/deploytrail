
import pandas as pd
data=pd.read_csv("https://raw.githubusercontent.com/AbhishekMali21/STUDENT-GRADE-ANALYSIS-PREDICTION/refs/heads/master/student-mat.csv")
print(data)

data.isnull().sum()

data.info()

data.columns

data[['Pstatus']]

data[['reason']]

data = data.drop(['address', 'sex', 'famsize', 'Mjob', 'Fjob', 'guardian', 'nursery', 'romantic', 'activities', 'higher'], axis=1)

data.columns

data = data.drop(['Pstatus'], axis=1)

data.dtypes

import sklearn

from sklearn.preprocessing import LabelEncoder
lb1=LabelEncoder()
data['school']=lb1.fit_transform(data['school'])

lb2=LabelEncoder()
data['reason']=lb2.fit_transform(data['reason'])

data[['schoolsup']]

lb3=LabelEncoder()
data['schoolsup']=lb3.fit_transform(data['schoolsup'])

lb4=LabelEncoder()
data['famsup']=lb3.fit_transform(data['famsup'])

lb5 = LabelEncoder()
data['paid'] = lb5.fit_transform(data['paid'])

lb6 = LabelEncoder()
data['internet'] = lb6.fit_transform(data['internet'])

data.columns

data.columns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


X = data.drop(['G3'], axis=1)
y = data['G3']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model=RandomForestRegressor()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# prompt: Give me a code that asks user to input the data and the model makes the prediction of the grade, MAKE SURE THAT YOU USE THE VAR NAMES FROM THE ABOVE CODE, I WANT NO EERORS, dont use the removed the columns - i removed dalc,walc, traveltime- i dont want the user to input numeric vaks only - ive label encoded

import pandas as pd # Import pandas for data manipulation


# Get user input for features
school = input("Enter school (0 for GP, 1 for MS): ")
reason = input("Enter reason (0 for course, 1 for home, 2 for other, 3 for reputation): ")
schoolsup = input("Enter schoolsup (0 for no, 1 for yes): ")
famsup = input("Enter famsup (0 for no, 1 for yes): ")
paid = input("Enter paid (0 for no, 1 for yes): ")
internet = input("Enter internet (0 for no, 1 for yes): ")
age = int(input("Enter age: "))
Medu = int(input("Enter mother's education (0-4): "))
Fedu = int(input("Enter father's education (0-4): "))
# traveltime = int(input("Enter travel time (1-4): ")) # This feature was removed from the training data, so we will not ask for it.
studytime = int(input("Enter study time (1-4): "))
failures = int(input("Enter number of past class failures: "))
famrel = int(input("Enter quality of family relationships (1-5): "))
freetime = int(input("Enter free time after school (1-5): "))
goout = int(input("Enter going out with friends (1-5): "))
health = int(input("Enter current health status (1-5): "))
absences = int(input("Enter number of school absences: "))
G1 = int(input("Enter grade for period 1: "))
G2 = int(input("Enter grade for period 2: "))


# Create a DataFrame with the user input
user_data = pd.DataFrame({
    'school': [int(school)],
    'reason': [int(reason)],
    'schoolsup': [int(schoolsup)],
    'famsup': [int(famsup)],
    'paid': [int(paid)],
    'internet': [int(internet)],
    'age': [age],
    'Medu': [Medu],
    'Fedu': [Fedu],
    'studytime': [studytime],
    'failures': [failures],
    'famrel': [famrel],
    'freetime': [freetime],
    'goout': [goout],
    'health': [health],
    'absences': [absences],
    'G1': [G1],
    'G2': [G2]
})

# List of features used during training the model
features = ['school', 'age', 'Medu', 'Fedu', 'reason', 'studytime',
            'failures', 'schoolsup', 'famsup', 'paid', 'internet', 'famrel',
            'freetime', 'goout', 'health', 'absences', 'G1', 'G2']

user_data = user_data[features]

predicted_grade = model.predict(user_data)

print(predicted_grade)

import pickle
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(lb1,open('lb1.pkl','wb'))
pickle.dump(lb2,open('lb2.pkl','wb'))
pickle.dump(lb3,open('lb3.pkl','wb'))
pickle.dump(lb4,open('lb4.pkl','wb'))
pickle.dump(lb5,open('lb5.pkl','wb'))
pickle.dump(lb6,open('lb6.pkl','wb'))