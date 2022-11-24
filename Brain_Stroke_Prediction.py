# IMPORT THE LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# READ THE DATASET
df = pd.read_csv('C:\\Users\\ranaw\\Downloads\\full_data.csv')
print(f'The dataset:\n{df.head()}')

# ANLAYSING THE DATASET
print(f'Random rowss:\n{df.sample(6)}')
print(f'\nThe brain stroke dataset includes the following\n: {df.columns}')
print(f'\nTotal Rows and Columns = {df.shape}')
print(f'\nStatistics: \n{df.describe()}')
print(f'\nChecking for NULL values: \n{df.isnull().sum()}')
print(f'\nData proportions: {df["gender"].unique()}')
print(f'{df["gender"].value_counts()}')
print ("\nMissing values: ", df.isnull().sum().values.sum())

# ANALYSING WITH VISUALISATION
# declaring data
stroke = df.loc[df['stroke']==1]

# PIE-PLOT
data = stroke['gender'].value_counts()
theLabels = ['Male', 'Female']
explode = [0, 0.1]
myColors = ['lightskyblue','lightpink']
# plotting data on chart
plt.pie(data, labels=theLabels,explode=explode,autopct='%.1f%%', colors=myColors, shadow=True)
plt.title('Gender Pie Chart (with stroke)')
plt.legend(loc='upper left')
plt.show()

# HISTOGRAM
sns.histplot(data=df,x='age',hue='stroke',palette='crest')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# SUBPLOTS
fig, axes = plt.subplots(2, 2, figsize=(12, 10), edgecolor = "black")
df.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0], edgecolor = "black")
df.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1], edgecolor = "black")
df.plot(kind='scatter', x='age', y='avg_glucose_level', color='green', ax=axes[1][0], title="Age vs. avg_glucose_level")
df.plot(kind='scatter', x='bmi', y='avg_glucose_level', color='red', ax=axes[1][1], title="bmi vs. avg_glucose_level")
plt.title('SUBPLOTS')
plt.show()

# HEATMAP
plt.figure(figsize=(10,10))
plt.title('Correlation Heatmap')
sns.heatmap(df.corr(),annot=True);
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.show()

# SMOKING STATUS
print(f'\nSmoking status of people with Brain stroke:\n{df["smoking_status"].unique()}')
smoking_status=['formerly smoked','never smoked','smokes','Unknown']
formerly_smoked=len(stroke[stroke['smoking_status']==smoking_status[0]])
never_smoked=len(stroke[stroke['smoking_status']==smoking_status[1]])
smokes=len(stroke[stroke['smoking_status']==smoking_status[2]])
unknown=len(stroke[stroke['smoking_status']==smoking_status[3]])
print('Never smoked:',never_smoked)
print('Formerly smoked:',formerly_smoked)
print('Smoked:',smokes)
print('Unknown:',unknown)
smoke_stroke = [formerly_smoked, never_smoked, smokes, unknown]

# BARPLOT
sns.barplot(y=smoke_stroke, x=smoking_status, palette='Pastel1')
plt.title("Stroke / Smoking Status")
plt.show()

# LABEL ENCODING
from sklearn.preprocessing import LabelEncoder 
lb = LabelEncoder()
# similar to smoking_status in IRIS dataset
df['ever_married'] = lb.fit_transform(df['ever_married'])
df['work_type'] = lb.fit_transform(df['work_type'])
df['Residence_type'] = lb.fit_transform(df['Residence_type'])
df['smoking_status'] = lb.fit_transform(df['smoking_status'])
df['gender'] = lb.fit_transform(df['gender'])

# DIVIDE DATASET INTO INDEPENDENT AND DEPENDENT VARIABLE
x = df.drop(['stroke'], axis=1)
y = df["stroke"]
print(f'y = \n{y}')

# SPLITTING x AND y INTO TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)
print(f"\nx_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# IMPORT THE MODEL/ALGORITHM - LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(max_iter=1000)

# TRAIN THE MODEL
lr.fit(x_train,y_train) 

# PREDICTION
lr_y_pred = lr.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, lr_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(lr_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(lr_y_pred, y_test)) 
print("r2_score = ", r2_score(lr_y_pred, y_test)) 
print("Accuracy of LOGISTIC REGRESSION MODEL in percentage (%): ", (accuracy_score(y_test, lr_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM - BERNOULLINB
from sklearn.naive_bayes import BernoulliNB 
bnb = BernoulliNB()

# TRAIN THE MODEL
bnb.fit(x_train,y_train) 

# PREDICTIONION
bnb_y_pred = bnb.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, bnb_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(bnb_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(bnb_y_pred, y_test)) 
print("r2_score = ", r2_score(bnb_y_pred, y_test)) 
print("Accuracy of NAIVE BAYES- BERNOULLI MODEL in percentage (%): ", (accuracy_score(y_test, bnb_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM - DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# TRAIN THE MODEL
dtc.fit(x_train,y_train) 

# PREDICTIONION
dtc_y_pred = dtc.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, dtc_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(dtc_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(dtc_y_pred, y_test)) 
print("r2_score = ", r2_score(dtc_y_pred, y_test)) 
print("Accuracy of DECISION TREE CLASSIFIER in percentage (%): ", (accuracy_score(y_test, dtc_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM - GAUSSIAN NB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# TRAIN THE MODEL
gnb.fit(x_train,y_train) 

# # PREDICTION
gnb_y_pred = gnb.predict(x_test) 
# print(f"First 5 actual values -\n{y_test.values[:5]},\nFirst 5 predicted values -\n{gnb_y_pred[:5]}")

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, gnb_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(gnb_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(gnb_y_pred, y_test)) 
print("r2_score = ", r2_score(gnb_y_pred, y_test)) 
print("Accuracy of NAIVE BAYES- GAUSSIAN MODEL in percentage (%): ", (accuracy_score(y_test, gnb_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()

# TRAIN THE MODEL
knn.fit(x_train,y_train) 

# PREDICTION
knc_y_pred = knn.predict(x_test) 
# print(f"First 5 actual values -\n{y_test.values[:5]},\nFirst 5 predicted values -\n{knc_y_pred[:5]}")

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, knc_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(knc_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(knc_y_pred, y_test)) 
print("r2_score = ", r2_score(knc_y_pred, y_test)) 
print("Accuracy of K- NEAREST NEIGHBOR CLASSIFIER in percentage (%): ", (accuracy_score(y_test, knc_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# TRAIN THE MODEL
rfc.fit(x_train,y_train) 

# PREDICTION
rfc_y_pred = rfc.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, rfc_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(rfc_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(rfc_y_pred, y_test)) 
print("r2_score = ", r2_score(rfc_y_pred, y_test)) 
print("Accuracy of RANDOM FOREST CLASSIFIER in percentage (%): ", (accuracy_score(y_test, rfc_y_pred))*100)

# IMPORT THE MODEL/ALGORITHM
from sklearn.svm import SVC
svm = SVC()

# TRAIN THE MODEL
svm.fit(x_train,y_train) 

# PREDICTION
svm_y_pred = svm.predict(x_test) 

# EVALUATION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , confusion_matrix, accuracy_score
print("\nMODEL EVALUATION: ")
print("Consfusion Matrixr-\n", confusion_matrix(y_test, svm_y_pred)) 
print("Mean_absolute_error = ", mean_absolute_error(svm_y_pred, y_test)) 
print("Mean_squared_error = ", mean_squared_error(svm_y_pred, y_test)) 
print("r2_score = ", r2_score(svm_y_pred, y_test)) 
print("Accuracy of SUPPORT VECTOR in percentage (%): ", (accuracy_score(y_test, svm_y_pred))*100)

# ---------------------PERFORMANCE ANALYSIS OF DIFFERENT MODELS -------------------
dataPerf = pd.DataFrame(data={'Model': ['LogisticRegression', 'BernoulliNB', 'Decision Tree Classifier', 'GaussianNB','K-Nearest Neighbours Classifier', 'Random Forest', 'SVM'], 'Score': [lr.score(x_test, y_test), bnb.score(x_test, y_test), dtc.score(x_test, y_test), gnb.score(x_test, y_test), knn.score(x_test, y_test), rfc.score(x_test, y_test), svm.score(x_test, y_test)]})

plt.figure(figsize=(12, 8))
sns.barplot(x="Model", y="Score", data=dataPerf, palette="magma")
plt.title('Performance analysis of different Models')
plt.show()