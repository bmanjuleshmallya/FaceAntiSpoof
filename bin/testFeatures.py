import os
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

featureName = 'LBP'
computeSVM = True

XTrainFilename = 'features/xTrain_{}.csv'.format(featureName)
XTestFilename = 'features/xTest_{}.csv'.format(featureName)
YTrainFilename = 'features/yTrain_{}.csv'.format(featureName)
YTestFilename = 'features/yTest_{}.csv'.format(featureName)

xTrain = pd.read_csv(XTrainFilename).values
xTest = pd.read_csv(XTestFilename).values

yTrain = pd.read_csv(YTrainFilename).values
yTest = pd.read_csv(YTestFilename).values

if not os.path.exists('results'):
    os.mkdir('results')

fileResults = 'results/{}.txt'.format(featureName)

f = open(fileResults, "w")

descriptorName = 'SVM Linear'
cValues = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000]
max_accuracy = 0
for cValue in cValues:
    print("Train {}".format(descriptorName))
    clfSVM = svm.SVC(C=cValue, kernel='linear', verbose=False, probability=True)
    clfSVM.fit(xTrain, yTrain)
    valuePredicted = clfSVM.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    if(accuracy > max_accuracy):
        max_accuracy = accuracy
        conf_matrix = confusionMatrix
        desc_name = descriptorName
    f.write('\n########################################\n')
    f.write("Train {}".format(desc_name))
    f.write('\nAccuracy: {}'.format(max_accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(conf_matrix)))

f.close()
