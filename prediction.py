import pandas as pd  
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import cv2
import numpy as np
import pandas as pd
import csv

image1 = cv2.imread('s1.jpg')
image2 = cv2.imread('s2.jpg')
image3 = cv2.imread('s3.jpg')

df = pd.read_csv('test.csv')
features = df[['area_0','area_1','area_2','area_3','area_4','area_5','area_6','area_7','area_8','area_9','area_10']].values
labels = df[['result']].values

rf=RandomForestClassifier()
rf.fit(features,labels)
pred_rf = rf.predict([[0, 26.0, 30.0, 21.0, 25.0, 10.0, 4387.5, 34.0, 4.0, 1.0, 4475.0]])
print("\n Random Forest_sample 1 : ",pred_rf)

pred_rf = rf.predict([[0, 36.5, 10.5, 61.5, 4146.0, 4176.5, 0, 0, 0, 0, 0]])
print("\n Random Forest_sample 2 : ",pred_rf)
print(" ")

pred_rf = rf.predict([[0, 4011.0, 4061.5, 0, 0, 0, 0, 0, 0, 0, 0]])
print("\n Random Forest_sample 2 : ",pred_rf)
print(" ")
print("using trainset and testset at 20% test data")
print(" ")

from sklearn.model_selection import train_test_split
from sklearn import metrics
X_trainset, X_testset, y_trainset, y_testset = train_test_split(features, labels, test_size=0.2, random_state=3)


print(" ")

rf=RandomForestClassifier(n_estimators = 120)
rf.fit(X_trainset,y_trainset)
pred_rf = rf.predict(X_testset)
#print (pred_rf [0:5])
#print (y_testset [0:5])
print("Random_Forest's Accuracy: ", metrics.accuracy_score(y_testset, pred_rf))
pred_rf = rf.predict([[0, 26.0, 30.0, 21.0, 25.0, 10.0, 4387.5, 34.0, 4.0, 1.0, 4475.0]])
'''print("\n Random Forest_sample 1 : ",pred_rf)
pred_rf = rf.predict([[0, 36.5, 10.5, 61.5, 4146.0, 4176.5, 0, 0, 0, 0, 0]])
print("\n Random Forest_sample 2 : ",pred_rf)
pred_rf = rf.predict([[0, 4011.0, 4061.5, 0, 0, 0, 0, 0, 0, 0, 0]])
print("\n Random Forest_sample 3 : ",pred_rf)'''

cv2.waitKey(0)
cv2.imshow("Original", image1)
cv2.waitKey(0)

cv2.imshow("Original", image2)
cv2.waitKey(0)

cv2.imshow("Original", image3)
cv2.waitKey(0)
cv2.destroyAllWindows()






