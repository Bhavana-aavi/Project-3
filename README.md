# Project-3
#MACHINE LEARNING TECHNIQUES IN PREDICTING DIABETES

import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from collections import Counter
df = pd.read_csv("/content/diabetes_data.csv")
df.head()
df.info()
df['gender'] = df['gender'].replace({'Male': 0, 'Female':1})
df.describe()
df.isna().sum()
df.shape
df.columns
plt.figure(figsize= (25,25))
p = sns.heatmap(df.corr(), annot = True, cmap= 'RdYlGn')
corr_mat = df.corr(method = 'pearson')
corr_mat
ls = list(df.corrwith(df['class']))
ls.sort(reverse= True)
ls[0:10]
df[df.columns[1:]].corr()['class'].sort_values(ascending=False)[:16]

x = df.drop(columns=['age','class','alopecia','itching'])
y= df.iloc[:, -1].values
Counter(y)
df['class'].value_counts().plot(kind= 'bar')
sns.countplot(y= df['gender'], hue=df['class'])
sns.countplot(y=df['polyuria'], hue= df['class'])
sns.countplot(y=df['polydipsia'], hue= df['class'])
sns.countplot(y=df['sudden_weight_loss'], hue= df['class'])
sns.countplot(y=df['weakness'], hue= df['class'])
sns.countplot(y=df['polyphagia'], hue= df['class'])
sns.countplot(y=df['visual_blurring'], hue= df['class'])
sns.countplot(y=df['genital_thrush'], hue= df['class'])
sns.countplot(y=df['irritability'], hue=df['class'])
sns.countplot(y=df['delayed_healing'], hue= df['class'])
sns.countplot(y=df['partial_paresis'], hue=df['class'])
sns.countplot(y=df['muscle_stiffness'], hue=df['class'])
sns.countplot(y=df['obesity'], hue=df['class'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state= 0, stratify = y)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state=0)
logReg.fit(x_train, y_train)
y_pred = logReg.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_proba = logReg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
def fit_and_evaluate_model(x_train, x_test, y_train, y_test, max_depth=5, min_samples_split=0.01, max_features=0.8, max_samples=0.8):
    random_forest = RandomForestClassifier(random_state=0,\
                                           max_depth = max_depth,\
                                           min_samples_split = min_samples_split,\
                                           max_features = max_features,
                                           max_samples=max_samples)
    model = random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confusion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:", random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
param_grid = [
    {'max_depth': [3,5,7,10], 'min_samples_split': [0.01, 0.03, 0.07, 0.1],
     'max_features': [0.7, 0.8, 0.9 ,1.0],
     'max_samples': [0.7, 0.8, 0.9, 1.0]}]
model = RandomForestClassifier()
search = GridSearchCV(estimator = model, param_grid = param_grid, cv=5, verbose=5)
search.fit(x_train, y_train)
results = pd.DataFrame(search.cv_results_)
results.sort_values('mean_test_score', inplace=True, ascending= False)
results.head(10)
results_save = pd.DataFrame(search.cv_results_)
results_save.to_csv("results_save.csv", index= False)
search.best_params_
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test, max_depth=5, min_samples_split=0.01, max_features=1.0, max_samples=1.0)
y_pred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc
Fpr, Tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.svm import SVC
model = SVC(kernel= "linear", C= 1.0, random_state=5)
SVC = model.fit(x_train, y_train)
pred= SVC.predict(x_test)
from sklearn.metrics import classification_report
print (classification_report(y_test, pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
print(roc_auc_score(y_test, pred))
print(confusion_matrix(y_test, pred))
Fpr, Tpr, _ = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

ttt= clf.fit(x_train, y_train)
y_predicted = ttt.predict(x_test)
y_predicted
print(classification_report(y_test, y_predicted))
print(accuracy_score(y_test, y_predicted))
print(roc_auc_score(y_test, y_predicted))
print(f1_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
Fpr, Tpr, _ = metrics.roc_curve(y_test, y_predicted)
auc = metrics.roc_auc_score(y_test, y_predicted)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train, Y_train = smt.fit_resample(x_train, y_train)
print('Before', Counter(y_train))
print('After', Counter(Y_train))
LR = LogisticRegression(random_state=1)
LR.fit(X_train, Y_train)
Y_pred = LR.predict(x_test)

CM = metrics.confusion_matrix(y_test, Y_pred)
print(CM)

target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, Y_pred, target_names=target_names))

#AUC

Y_pred_proba = LR.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_proba)
auc = metrics.roc_auc_score(y_test, Y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
def fit_and_evaluate_model(X_train, x_test, Y_train, y_test, max_depth=5, min_samples_split=0.01, max_features=0.8, max_samples=0.8):
    random_forest = RandomForestClassifier(random_state=0,\
                                           max_depth = max_depth,\
                                           min_samples_split = min_samples_split,\
                                           max_features = max_features,
                                           max_samples=max_samples)
    model = random_forest.fit(X_train, Y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confusion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:", random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model
model_Smote = fit_and_evaluate_model(X_train, x_test, Y_train, y_test)
param_grid_Smote = [
    {'max_depth': [3,5,7,10], 'min_samples_split': [0.01, 0.03, 0.07, 0.1],
     'max_features': [0.7, 0.8, 0.9 ,1.0],
     'max_samples': [0.7, 0.8, 0.9, 1.0]}]
model = RandomForestClassifier()
search = GridSearchCV(estimator = model_Smote, param_grid = param_grid_Smote, cv=5, verbose=5)
search.fit(X_train, Y_train)
results = pd.DataFrame(search.cv_results_)
results.sort_values('mean_test_score', inplace=True, ascending= False)
results.head(10)
results_save = pd.DataFrame(search.cv_results_)
results_save.to_csv("results_save.csv", index= False)
search.best_params_
model = fit_and_evaluate_model(X_train, x_test, Y_train, y_test, max_depth=5, min_samples_split=0.01, max_features=1.0, max_samples=0.9)
y_pred_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc
Fpr, Tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.svm import SVC
model = SVC(kernel= "linear", C= 1.0, random_state=5)
SVC = model.fit(X_train, Y_train)
pred= model.predict(x_test)
pred
print (classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(roc_auc_score(y_test, pred))
print(confusion_matrix(y_test, pred))
Fpr, Tpr, _ = metrics.roc_curve(y_test, pred)
auc = metrics.roc_auc_score(y_test, pred)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
NB_smote= clf.fit(X_train, Y_train)
y_predicted = clf.predict(x_test)
y_predicted
print(classification_report(y_test, y_predicted))
print(accuracy_score(y_test, y_predicted))
print(roc_auc_score(y_test, y_predicted))
print(f1_score(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))
Fpr, Tpr, _ = metrics.roc_curve(y_test, y_predicted)
auc = metrics.roc_auc_score(y_test, y_predicted)
plt.plot(Fpr,Tpr, label = "data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
