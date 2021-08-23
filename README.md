# FaceAntispoof
Detect spoofing attacks

## About
#### Referenced from [imironica/liveness](https://github.com/imironica/liveness)
#### Dataset used: [NUAA dataset](https://drive.google.com/file/d/1-aSGKdAIK0YoKxQvnNx1KJvTm4zbwZLz/view)
**Note**: Read the [usage agreement](http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html) first before downloading.

## Preparation
### How to place the dataset?
Just drag the *raw* file in the directory.

### Modifications
The in-built text files (**client_test_raw.txt**, **client_train_raw.txt**, **imposter_test_raw.txt** and **imposter_train_raw.txt**) needs to have the **backslash(\\)** replaced for all files, with a **forward slash(/)**.

## What has been added:
- *Histogram of Oriented Gradients (HOG) face detector* for preprocessing dataset (will crop faces much more accurately).
- Feature extraction with *Local Binary Patter (LBP)* (will also generate a new dataset in csv format).
- Added *Support Vector Machines (SVM)* 
- Added implementation for preprocessing web-camera data .

## Notes:
- We may have to remove HOG for web-camera face detection, as it is very slow, and we can see a drop in the frame rate. Since a video is a continuous frame of images, we might as well try to implement face crop using Haar Cascade.

