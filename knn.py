import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import cross_val_score(unable to import)
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier  #takes distance explicitly when making predictions
from sklearn.preprocessing import Imputer #used when some data is missing
from sklearn.preprocessing import scale  #used to scale the data(normalization)
from sklearn.preprocessing import StandardScaler  #scaling in pipeline
from sklearn.pipeline import Pipeline #imputing within a pipeline
from sklearn.svm import SVC   #support vector classification

plt.style.use('ggplot')

iris = datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data),type(iris.target)
iris.data.shape
iris.target_names
X = iris.data 
y = iris.target
df = pd.DataFrame(X,columns = iris.feature_names)
print(df.head())
_ = pd.scatter_matrix(df,c = y,figsize=[8,8],s=150,marker='D')

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X,y)
y_pred = knn.predict(X)
new_prediction = knn.predict(X)
print("Prediction: {}".format(new_prediction))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)
knn.score(X_test,y_test)


reg = linear_model.LinearRegression()
reg.fit(X,y)

#Prediction_space = np.linspace(min(X),max(X)).reshape(-1,1)
#confusion_matrix()

logreg = LogisticRegression()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=21)
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr
	)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

roc_auc_score(y_test,y_pred_prob)
cv_scores = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')
print(cv_scores)

#hyperparameter

param_grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)
knn_cv.best_params_

knn_cv.best_score_


#using imputers when some data is missing in the datasets

imp = Imputer(missing_values='NaN',strategy='mean',axis=0)  #most_frequent can also be used instead of mean
imp.fit(X)
X = imp.transform(X)
logreg = LogisticRegression()
steps = [('imputation',imp),('logistic_regression',logreg)]
pipeline = Pipeline(steps)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
pipeline.score(X_test,y_test)

"""df[df == '?'] = np.nan
-> Print the number of NaNs
print(df.isnull().sum())
 ->Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))
 ->Drop missing values and print shape of new DataFrame
df = df.dropna()
"""


#scaling in scikit
X_scaled = scale(X)
np.mean(X),np.std(X)
np.mean(X_scaled) , np.std(X_scaled)

#scaling in pipeline
steps = [('scaler',StandardScaler()),('knn',KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)
knn_scaled = pipeline.fit(X_train,y_tain) #fit the pipeline in training set
y_pred = pieline.predict(X_test)
accuracy_score(y_test,y_pred)

#knn without scale
knn_unscaled = KNeighborsClassifier().fit(X_train,y_train)
knn.unscaled.score(X_test,y_test)

#cross validation and scaling in a pipeline

steps = [('scaler',StandardScaler()),('knn',KNeighborsClassifier())]
pipeline=Pipeline(steps)
parameters={'knn__n_neighbors':np.arange(1,50)}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)
cv = GridSearchCV(pipeline,param_grid=parameters)
cv.fit(X_train,y_train)
cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test,y_test))
print(classification_report(y_test,y_pred))











