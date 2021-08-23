import cv2
import pandas as pd
import scipy.misc
import os
import dlib

def readDataset():
    lstFileTrainValid = 'raw/client_train_raw.txt'
    lstFileTrainImposter = 'raw/imposter_train_raw.txt'
    lstFileTestValid = 'raw/client_test_raw.txt'
    lstFileTestImposter = 'raw/imposter_test_raw.txt'

    dfTrainValid = pd.read_csv(lstFileTrainValid, header=None, names=['Path'])
    dfTrainValid['Path'] = 'raw/ClientRaw/' + dfTrainValid['Path']
    dfTrainValid['Label'] = 1

    dfTrainImposter = pd.read_csv(lstFileTrainImposter, header=None, names=['Path'])
    dfTrainImposter['Path'] = 'raw/ImposterRaw/' + dfTrainImposter['Path']
    dfTrainImposter['Label'] = 0

    dfTestValid = pd.read_csv(lstFileTestValid, header=None, names=['Path'])
    dfTestValid['Path'] = 'raw/ClientRaw/' + dfTestValid['Path']
    dfTestValid['Label'] = 1

    dfTestImposter = pd.read_csv(lstFileTestImposter, header=None, names=['Path'])
    dfTestImposter['Path'] = 'raw/ImposterRaw/' + dfTestImposter['Path']
    dfTestImposter['Label'] = 0

    dfTrain = pd.concat([dfTrainValid, dfTrainImposter])
    dfTest = pd.concat([dfTestValid, dfTestImposter])
    return dfTrain, dfTest

def makeDir():
    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    if not os.path.exists('dataset/train'):
        os.mkdir('dataset/train')
    if not os.path.exists('dataset/train/0'):
        os.mkdir('dataset/train/0')
    if not os.path.exists('dataset/train/1'):
        os.mkdir('dataset/train/1')
    if not os.path.exists('dataset/test'):
        os.mkdir('dataset/test')
    if not os.path.exists('dataset/test/0'):
        os.mkdir('dataset/test/0')
    if not os.path.exists('dataset/test/1'):
        os.mkdir('dataset/test/1')


def writeFile(df, write_dir):
    invalid=0
    no_faces=0
    hog_detector = dlib.get_frontal_face_detector()
    for index, row in df.iterrows():
        no_faces +=1
        imagePath = str(row['Path'])
        label = str(row['Label'])
        img = dlib.load_rgb_image(imagePath)
        newPath = write_dir+'{}/{}.jpg'.format(label, index)
        print(newPath)
        dets = hog_detector(img, 1)
        if(len(dets)==0):
            invalid+=1
            print("Unrecognized face {}".format(imagePath))
            continue
        
        for i, d in enumerate(dets):
            crop = img[d.top():d.bottom(), d.left():d.right()]
            if crop.size==0:
                print("Ignored this - crop picture seems to be empty")
            else:
                cv2.imwrite(newPath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print("--- {}/{} Total unrecognized faces".format(invalid, no_faces))

'''
Run this function to generate faces. This is what you have to import.
'''
def generateCropImages():
    dfTrain, dfTest = readDataset()
    makeDir()
    writeFile(dfTrain, 'dataset/train/')
    writeFile(dfTest, 'dataset/test/')