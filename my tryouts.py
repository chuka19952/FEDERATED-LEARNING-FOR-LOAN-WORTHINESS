import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series,DataFrame 
from datetime import date
import datetime as DT
import io


#importing the datasets
df12=pd.read_csv('train.csv')
df13=pd.read_csv('test.csv')


#DATA PRE-PROCESSING
#for train set
df12['Gender'].value_counts()
df12.Gender = df12.Gender.fillna('Male')

df12['Married'].value_counts()
df12.Married = df12.Married.fillna('Yes')

df12['Dependents'].value_counts()
df12.Dependents = df12.Dependents.fillna('0')

df12['Self_Employed'].value_counts()
df12.Self_Employed = df12.Self_Employed.fillna('No')

df12['LoanAmount'].value_counts()
df12.LoanAmount = df12.LoanAmount.fillna(df12['LoanAmount'].mean())

df12['Loan_Amount_Term'].value_counts()
df12.Loan_Amount_Term = df12.Loan_Amount_Term.fillna(360)

df12['Credit_History'].value_counts()
df12.Credit_History = df12.Credit_History.fillna(1.0)

#for test set
df13['Gender'].value_counts()
df13.Gender = df13.Gender.fillna('Male')

df13['Married'].value_counts()
df13.Married = df13.Married.fillna('Yes')

df13['Dependents'].value_counts()
df13.Dependents = df13.Dependents.fillna('0')

df13['Self_Employed'].value_counts()
df13.Self_Employed = df13.Self_Employed.fillna('No')

df13['LoanAmount'].value_counts()
df13.LoanAmount = df13.LoanAmount.fillna(df12['LoanAmount'].mean())

df13['Loan_Amount_Term'].value_counts()
df13.Loan_Amount_Term = df13.Loan_Amount_Term.fillna(360)

df13['Credit_History'].value_counts()
df13.Credit_History = df13.Credit_History.fillna(1.0)


#we now encode our categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
df12["Loan_ID"] = labelencoder_X.fit_transform(df12["Loan_ID"])


labelencoder_X = LabelEncoder()
df12["Gender"] = labelencoder_X.fit_transform(df12["Gender"])

labelencoder_X = LabelEncoder()
df12["Dependents"] = labelencoder_X.fit_transform(df12["Dependents"])

labelencoder_X = LabelEncoder()
df12["Married"] = labelencoder_X.fit_transform(df12["Married"])

labelencoder_X = LabelEncoder()
df12["Education"] = labelencoder_X.fit_transform(df12["Education"])

labelencoder_X = LabelEncoder()
df12["Self_Employed"] = labelencoder_X.fit_transform(df12["Self_Employed"])

labelencoder_X = LabelEncoder()
df12["Property_Area"] = labelencoder_X.fit_transform(df12["Property_Area"])

labelencoder_X = LabelEncoder()
df12["Loan_Status"] = labelencoder_X.fit_transform(df12["Loan_Status"])





#encoding categorical variables for test.csv
labelencoder_X = LabelEncoder()
df13["Loan_ID"] = labelencoder_X.fit_transform(df13["Loan_ID"])


labelencoder_X = LabelEncoder()
df13["Gender"] = labelencoder_X.fit_transform(df13["Gender"])


labelencoder_X = LabelEncoder()
df13["Married"] = labelencoder_X.fit_transform(df13["Married"])


labelencoder_X = LabelEncoder()
df13["Education"] = labelencoder_X.fit_transform(df13["Education"])

labelencoder_X = LabelEncoder()
df13["Self_Employed"] = labelencoder_X.fit_transform(df13["Self_Employed"])

labelencoder_X = LabelEncoder()
df13["Property_Area"] = labelencoder_X.fit_transform(df13["Property_Area"])

labelencoder_X = LabelEncoder()
df13["Dependents"] = labelencoder_X.fit_transform(df13["Dependents"])




"""#splitting our dataset in X and y
X_train=df12.iloc[:, 1:12].values
y_train=df12.iloc[:, 12:].values
X_test=df13.iloc[:, 1:].values"""

#splitting our dataset 2
X=df12.iloc[:, 1:12].values
y=df12.iloc[:, 12:].values




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#one-hot encoding "property area" for X_train
onehotencoder = OneHotEncoder(categorical_features = [10])
X_train = onehotencoder.fit_transform(X_train).toarray()
#dropping one categorical var column to avoid categorical var trap
X_train=X_train[:, 1:]


#one-hot encoding "property area" for X_test
onehotencoder = OneHotEncoder(categorical_features = [10])
X_test = onehotencoder.fit_transform(X_test).toarray()
#dropping one categorical var column to avoid categorical var trap
X_test=X_test[:, 1:]



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)    ----scalling y_train would affect our outcome and wont even fit


                              #APPLYING KNN
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
""" Finally, usinig the KNN, we had an accuraccy of 64% and 83% with just 4 features"""
    
                          #APPLYING LOGISTIC REGRESSION
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
""" And logistics regression gave us an accuracy of 84%--the best so far"""

                         #APPLYING DESCISION TREE
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
""" and descision tree gave us 74%"""

                     #APPLYING NAIVE BAYES
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
""" WE got 83% with the naive bayes """

                      #APPLYING K-MEANS
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
y_pred = kmeans.fit_predict(X_test)
""" k-means gave us just 60% even with pca"""

                   # APPLYING RANDOM FOREST
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
""" RF gave us 73%"""

               #APPLYING KERNEL SVM
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""and svm yeilds 83%"""




# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X_train, y_train)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_train)
# summarize selected features
print(features[0:5,:])





# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X_train, y_train)
print("Num Features: %d")  #%fit.n_features_
print("Selected Features: %s") # %fit.support_
print("Feature Ranking: %s")  #%fit.ranking_



# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
print(model.feature_importances_)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#goto paper with code and check the sota(state of the art) on facenet, arcface, deep face and open face
#mask rcnn for image segmentation or unet


# SYNTAX/CODE FOR FEDERATED LEARNING 
trainer = tff.learning.build_federated_averaging_process(...)
state = trainer.initialize()
federated_training_data = ...

def sample(federate_data):
  return ...

while True:
  data_for_this_round = sample(federated_training_data)
  state, metrics = trainer.next(state, data_for_this_round)

