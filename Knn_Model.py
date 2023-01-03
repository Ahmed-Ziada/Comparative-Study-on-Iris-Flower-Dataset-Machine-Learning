# Step1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Step2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Step3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Step4
from sklearn.metrics import confusion_matrix
# Step5
from matplotlib.colors import ListedColormap


from sklearn.feature_selection import SelectKBest, f_classif

from numpy import mean
from sklearn.model_selection import cross_val_score


















# ============================== Step1 Import Dataset
dataset = pd.read_csv(r'..\Dataset\Data set 1 (5 KB) - iris.csv.csv')
print(dataset.head())

# Feature Data
X = dataset.iloc[:,[1,2,3,4]].values
#SepalLengthCm,[SepalWidthCm],PetalLengthCm,PetalWidthCm

# Target Data
y = dataset.iloc[:,5].values #Species


# ============================== Step 2 Data Splitting


# we can apply feature selection to Select features according to the k highest scores
bestfeatures = SelectKBest(score_func=f_classif, k=3)
iris_trim = bestfeatures.fit_transform(X, y)
print(bestfeatures.scores_)
#SepalWidthCm has the least score so we can remove it from the features    
X = dataset.iloc[:,[1,3,4]].values


# Spliting dataset into features traning set(X_train , X_test) and 
# target test set(y_train , y_test)
X_train, X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.25 , 
random_state = 42)


#Scalling and Standardization
#normalize/standardize i.e. μ = 0 and σ = 1 your features/variables/columns of X, 
# individually, before applying any machine learning model.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# ============================== Step 3 Train the Model
# Number of Neighbors
k_range = range(1,26)

scores = {} #dictionary {key(k):value(accuracy)}
scores_list = []

for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors= k)
    classifier.fit(X_train , y_train)

    y_pred = classifier.predict(X_test)
    #this function computes subset accuracy: 
    # the set of labels predicted for a sample must exactly 
    # match the corresponding set of labels in y_true
    scores[k]= accuracy_score(y_test,y_pred)   
    
    scores_list.append(accuracy_score(y_test,y_pred))


#plot relation between k and test accuracy (scores_list)
plt.plot(k_range , scores_list)
plt.xlabel = ('Value of K for KNN')
plt.ylabel = ('Testing Accuracy')
plt.show()



# so we choose 8 as our k_value
n_neighbors = 8
classifier = KNeighborsClassifier(n_neighbors)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# ============================== Step 4  Evaluate the model
cm = confusion_matrix(y_test,y_pred)
cm1 = accuracy_score(y_test,y_pred)

print('Confusion Matrix of KNN : \n', cm)
print('Accuracy of KNN:','% 0.2f' % cm1)



sc = mean(cross_val_score(classifier, X, y, cv=10))
print('Cross Validation Score of KNN : ',sc)


# ============================== Step 5  Plot the model
#we take only the last 2 features since they are the best features
X = dataset.iloc[:,[3,4]].values 

d = {'Iris-setosa' : 0,'Iris-versicolor' : 1, 'Iris-virginica' : 2}
dataset['Species'] = dataset['Species'].map(d)
y = dataset.iloc[:,5].values #Species
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Iris-Flower classification (k = %i)"
            % (n_neighbors))
plt.show()   


