import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import local_binary_pattern

def computeFeatures(X, Y, currentFolder):
    i = 0
    folders = ['0', '1']
    for folder in folders:
        images = [os.path.join(currentFolder, folder, file) for file in os.listdir(currentFolder    +folder)]

        for path in images:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, 24, 8, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 24+3), range=(0, 24+2))
            hist = hist/sum(hist)
            X.append(hist)
            Y.append(int(folder))

            if i%100 == 0:
                print(i)
            i+=1

def computeLBP():
    folderImages = 'dataset/{}/'
    
    print("Training dataset")
    Xtrain = []
    Ytrain = []
    currentFolder = folderImages.format('train')
    print('Computing train features')
    computeFeatures(Xtrain, Ytrain, currentFolder)
    
    print("Training dataset")
    Xtest = []
    Ytest = []
    currentFolder = folderImages.format('train')
    print('Computing test features')
    computeFeatures(Xtest, Ytest, currentFolder)
    
    if not os.path.exists('features'):
        os.mkdir('features')
    
    df = pd.DataFrame(Xtrain)
    df.to_csv("features/xTrain_LBP.csv", index=False)
    
    df = pd.DataFrame(Xtest)
    df.to_csv("features/xTest_LBP.csv", index=False)
    
    df = pd.DataFrame(Ytrain)
    df.to_csv("features/yTrain_LBP.csv", index=False)
    
    df = pd.DataFrame(Ytest)
    df.to_csv("features/yTest_LBP.csv", index=False)
