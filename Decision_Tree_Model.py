import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree


from sklearn.feature_selection import SelectKBest, f_classif


from sklearn.model_selection import cross_val_score
from numpy import mean




from sklearn.metrics import precision_score


# ============================== Step1 Import Dataset
dataset = pd.read_csv(r'..\Dataset\Data set 1 (5 KB) - iris.csv.csv')
print(dataset.head())


# Feature Data
X = dataset.iloc[:,[1,2,3,4]].values


# Target Data
y = dataset.iloc[:,5].values



# ============================== Step 2 Data Splitting

# we can apply feature selection to Select features according to the k highest scores
bestfeatures = SelectKBest(score_func=f_classif, k=3)
iris_trim = bestfeatures.fit_transform(X, y)
print(bestfeatures.scores_)
#SepalWidthCm has the least score so we can remove it from the features    
X = dataset.iloc[:,[1,3,4]].values


# Spliting dataset into features traning set(X_train , X_test) and 
# target test set(y_train , y_test)
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.25, random_state=42)

#Scalling and Standardization
#normalize/standardize i.e. μ = 0 and σ = 1 your features/variables/columns of X, individually, before applying any machine learning model.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ============================== Step 3 Train the Model
#Sklearn supports “gini” criteria for Gini Index and by default, it takes “gini” value.
clf= DecisionTreeClassifier(criterion  = 'gini') #or entropy(0.97) 
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)



# ============================== Step 4  Evaluate the model
cm = confusion_matrix(y_test,y_pred)
cm1 = accuracy_score(y_test,y_pred)

print('Confusion Matrix of Descion Tree : \n', cm)
print('Accuracy of Descion Tree : ','% 0.2f' % cm1)




sc = mean(cross_val_score(clf, X, y, cv=10))
print('Cross Validation Score of Decsion Tree : ',sc)

# ============================== Step 5  Plot the model
features = dataset.columns[1:5].values
features = list(features)
target =  ['setosa', 'versicolor', 'virginica']

fig = plt.figure(figsize=(10,15))
nodes = tree.plot_tree(clf, feature_names=features, class_names=target,filled=True)
plt.show()


