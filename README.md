# Captcha kapor solver

## download.sh

Allows to download captchas from katasterportal.sk

```
./download.sh 200
```
Downloads 200 captcha images and stores them in `unlabeled` directory.

## annotate.py

Tool to manually annotate downloaded images in `unlabeled` directory.

```
python annotate.py
```

## trainAndTest.py
katasterportal.sk is using naive captcha system:
![Captcha example](/docs/1.jpg)
![Captcha example 2](/docs/2.jpg)

### Things that I gather:
- captcha contains 5 characters (letters + digits)
- background is gradient from blue to white
- characters are always black
- font is always the same
- no geometric deformations
- double crossed with grey line 

### Solution
I wanted to be robust again color change and number of characters so I use simple opencv filters:
- convert to greyscale and then treshold to binary image to remove background
- use combination of erode and dilate to remove horizontal grey line 
- find connected components and consider them individual characters
- execute machine learning to classify component. 

### Machine learning part
Each component is grid of size 35x30(rows x cols). Then this vector is classified using:
- k-nearest neighbour
- linear svm - multi class version

### Experiments
I manually annotated 200 captchas what should be 1000 charactes to classify. 80% of captchas are used as training data. 

```
Train skipped: 10
Test skipped: 4
X = (750, 1050)
Y = (750,)
======= Model: knn
Test error: 0.0
Erorr abs: 0 / 180
===========
SVM coef:
(22, 1050)
======= Model: linSVM
Test error: 0.0
Erorr abs: 0 / 180
===========
```

## Conclussion
[Katasterportal.sk](https://www.katasterportal.sk/kapor/) is using fairly ineffective captcha system. If separation of characters from image is successful then test error is very near 0. Separation of characters failed in 14 from 200 cases = 7%, which I believe can be pushed down if needed and if one submition fails you can always try several time to break captcha. 


Run nearest neighbour and svm to classify isolated letters. 
