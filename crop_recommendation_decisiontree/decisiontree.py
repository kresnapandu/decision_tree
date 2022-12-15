import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Pengambilan dan persiapan data
data = pd.read_csv('Crop_recommendation.csv')
le = LabelEncoder()

#Preprocessing
data['label'] = le.fit_transform(data['label'])

x = data[['temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

#Training data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 0)

# Model Decision Tree 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree = tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_tree)
print('Akurasi Decision tree (train): ', "{:.0%}". format(accuracy)) 
print('Akurasi Decision (test): ', "{:.0%}". format(accuracy_score(y_train, tree.predict(x_train)))) 
print(20*'--')  

# Model KNN 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier()
knn = knn.fit(x_train, y_train) 
y_pred_knn = knn.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred_knn)
print('Akurasi KNN (train): ', "{:.0%}". format(accuracy)) 
print('Akurasi KNN (test): ', "{:.0%}". format(accuracy_score(y_train, knn.predict(x_train)))) 
print(20*'--')  

# Model SVM
from sklearn.svm import SVC 
svc = SVC()
svc = svc.fit(x_train, y_train) 
y_pred_svc = svc.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred_svc)
print('Akurasi SVM (train): ', "{:.0%}". format(accuracy)) 
print('Akurasi SVM (test): ', "{:.0%}". format(accuracy_score(y_train, svc.predict(x_train)))) 
print(20*'--')

#Cross Validation
from sklearn.model_selection import cross_val_score
cvTree = cross_val_score(tree, x, y, cv=20)
cvKNN = cross_val_score(knn, x, y, cv=20)
cvSVM = cross_val_score(svc, x, y, cv=20)

print("Tree Cross Validation Score: %f", cvTree.mean())
print("KNN Cross Validation Score: %f", cvKNN.mean())
print("SVM Cross Validation Score: %f", cvSVM.mean())