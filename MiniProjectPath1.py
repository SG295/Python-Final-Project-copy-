import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import ConfusionMatrixDisplay

''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))

print(dataset_1.to_string()) #This line will print out your data

# -_-_-_- Question 1 -_-_-_-
# Linear Regression with varying groups of three
"""brooklyn = (np.array(dataset_1))[:,5] #starts w/ 1704
manhattan = (np.array(dataset_1))[:,6] #starts w/ 3126
williamsburg = (np.array(dataset_1))[:,7] #starts w/ 4115
queensboro = (np.array(dataset_1))[:,8] #starts w/ 2552"""
brook = dataset_1.loc[::, "Brooklyn Bridge"]
manhattan = dataset_1.loc[::, "Manhattan Bridge"]
williamsburg = dataset_1.loc[::, "Williamsburg Bridge"]
queensboro = dataset_1.loc[::, "Queensboro Bridge"]
total = dataset_1.loc[::, "Total"]
total = [x.replace(",", "") for x in total]
total = np.array(total)
all_bridges = [brook, manhattan, williamsburg, queensboro]
all_names = {brook.name, manhattan.name, williamsburg.name, queensboro.name}

#Loop to do multiple linear regression for each group of three bridges (leaving a new one out each time).
#Fill dict with bridge names in model as key, and value as results such as MSE, r2, and model coefs and intercepts
i = 0 #index variable
models = {} #dict with which bridges were included in model as key
while i < 4:
    names = []
    test_list = all_bridges.copy()
    del test_list[i]
    for j in test_list:
        names.append(j.name)
    x = dataset_1[names]
    y = total
    X_train, X_test, y_train, y_test = train_test_split(x, total, test_size = .2, random_state=0)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    #print('Intercepts: \n', regr.intercept_)
    #print('Coefficients: \n', regr.coef_)

    coeff1 = []
    coeff2 = []
    coeff3 = []
    y_pred = []
    for k in (list(X_test[names[0]])):
        output = ((regr.coef_[0]) * k)
        coeff1.append(output)
    for l in (list(X_test[names[1]])):
        output = ((regr.coef_[1]) * l)
        coeff2.append(output)
    for m in (list(X_test[names[2]])):
        output = ((regr.coef_[2]) * m)
        coeff3.append(output)

    n = 0
    while n < len(X_test[names[0]]):
        output = coeff1[n] + coeff2[n] + coeff3[n] + regr.intercept_
        y_pred.append(output)
        n += 1

    y_test = [int(x) for x in y_test]
    y_pred = [math.trunc(x) for x in y_pred]
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    
    r_squared = r2_score(y_test, y_pred)

    results = [regr.intercept_, regr.coef_, mse, r_squared]

    models[tuple(names)] = results

    i += 1

print(models)

#Fills list of all model names, r2, and MSE as well as prints each models results to determine the best one
model_name = []
all_r2 = []
all_mse = []

for i in models:
    print(f"The model for the {i[0]}, {i[1]}, and {i[2]} has an MSE of {(models[i])[2]:.3f} and an R-squared value of {(models[i])[3]:.3f}.")
    model_name.append(i)
    all_mse.append((models[i])[2])
    all_r2.append((models[i])[3])
    print("\n")

#Prints best model and its MSE and R-squared values
print(f"The model with the lowest MSE of {min(all_mse):3f} is the one with the {(model_name[all_mse.index(min(all_mse))])[0]}, {(model_name[all_mse.index(min(all_mse))])[1]}, and {(model_name[all_mse.index(min(all_mse))])[2]}.")
print("\n")
print(f"The model with the maximum R-squared of {max(all_r2):3f} is the one with the {(model_name[all_r2.index(max(all_r2))])[0]}, {(model_name[all_r2.index(max(all_r2))])[1]}, and {(model_name[all_r2.index(max(all_r2))])[2]}.")
print("\n")
best_bridges = set(model_name[all_mse.index(min(all_mse))])
worst_bridge = all_names.difference(best_bridges)
worst_bridge = list(worst_bridge)
print(f"Thus, the best model is the {(model_name[all_mse.index(min(all_mse))])[0]}, {(model_name[all_mse.index(min(all_mse))])[1]}, and {(model_name[all_mse.index(min(all_mse))])[2]} model, and the bridge we should NOT use a sensor on is the {worst_bridge[0]}.")
print("\n")

#Prints final model 
print(f"Model: Total Traffic = {(models[(('Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'))])[1][0]:.3f} * {(model_name[all_mse.index(min(all_mse))])[0]} Traffic + {(models[(('Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'))])[1][1]:.3f} * {(model_name[all_mse.index(min(all_mse))])[1]} + {(models[(('Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'))])[1][2]:.3f} * {(model_name[all_mse.index(min(all_mse))])[2]} + {(models[(('Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge'))])[0]:.3f}")
print("\n")



# -_-_-_- Question 2 -_-_-_-
low_temp = dataset_1.loc[::, 'Low Temp']
high_temp = dataset_1.loc[::, 'High Temp']
precipitation = dataset_1.loc[::, 'Precipitation']
total = dataset_1.loc[::, 'Total']
total = [x.replace(",", "") for x in total]
total = np.array(total)

X = np.transpose(np.matrix([low_temp, high_temp, precipitation]), axes=None)
total = total.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, total, test_size = .2, random_state=0)

#Uses LazyRegressor to find best regression model for data
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
print("\n\n")
print(f"The best classifier model is {((models.iloc[0]).name)} with an adjusted R-squared of {((models.iloc[0])[0]):.3f}, a R-squared of {((models.iloc[0])[1]):.3f}, and a root mean square error (RMSE) of {((models.iloc[0])[2]):.3f}.")
print("\n\n")

#Uses determined best regression model to make a model/predict data
model = PassiveAggressiveRegressor()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
predict = model.predict(X_test)
data = pandas.DataFrame(data={"Predictions": predict.flatten(), "Actual": y_test, "Percent Error": abs(((predict - y_test)/y_test)*100)})
print(data)



# -_-_-_- Question 3 -_-_-_-
days = (np.array(dataset_1))[:,1] #Friday start

X_train, X_test, y_train, y_test = train_test_split(total, days, test_size = .2, random_state=0) #test_size=0.33, random_state=42)
num_class = 7 #number days of week

#LazyClassifier tests all classifier models to find the best one for our data
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)


print(models)

print("\n\n")
print(f"The best classifier model is {((models.iloc[0]).name)} with an accuracy of {((models.iloc[0])[0]):.3f} and a balanced accuracy of {((models.iloc[0])[1])}.")
print("\n\n")

#Uses chosen best classifier model to predict days and create a confusion matrix
model = LinearDiscriminantAnalysis()
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
model.fit(X_train, y_train)
predict = model.predict(X_test)
confusion = confusion_matrix(predict, y_test, labels=['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])
print("Confusion Matrix:")
print(confusion)
print("\n\n")
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels = ['Fri', 'Sat', 'Sun', 'Mon', 'Tues', 'Wed', 'Thur'])


# -_-_-_- Plotting -_-_-_-

#Initial Data Plotting
x = np.linspace(1, len(brook), len(brook))
fig, ax = plt.subplots()
ax.plot(x, brook, label='Brooklyn Bridge')
ax.plot(x, manhattan, label='Manhattan Bridge')
ax.plot(x, williamsburg, label='Williamsburg')
ax.plot(x, queensboro, label='Queensboro Bridge')
ax.set_xlabel('Days')
ax.set_ylabel('Traffic on Bridge')
ax.set_title('Bridge Traffic')
ax.legend()
plt.show()

plt.plot(x, total, label="Total Traffic")
plt.xlabel('Days')
plt.ylabel('Traffic on Bridge')
plt.title('TOTAL Bridge Traffic')
plt.show()

#Q1 Plotting
day = np.linspace(1, len(y_test), len(y_test))
fig1, ax1 = plt.subplots()
ax1.scatter(day, y_test, label='Actual Total Traffic')
ax1.scatter(day, y_pred, label='Predicted Total Traffic')
ax1.set_xlabel('Day')
ax1.set_ylabel('Total Traffic')
ax1.set_title('Predicted Traffic for Best Multiple Regression')
ax1.legend()
plt.show()

#Q2 Plotting
day = np.linspace(1, len(y_test), len(y_test))
fig2, ax2 = plt.subplots()
ax2.scatter(day, y_test, label='Actual Total Traffic')
ax2.scatter(day, predict, label='Predicted Total Traffic')
ax2.set_xlabel('Day')
ax2.set_ylabel('Total Traffic')
ax2.set_title('Predicted Traffic for Passive Aggressive Regressor')
ax2.legend()
plt.show()

#Q3 Confusion Matrix
cm_display.plot()
plt.show()