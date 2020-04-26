
import numpy as np
import cv2 as cv
import os

def clearImage(img):
  margin = 2
  img = img[margin:-margin,margin:-margin]
  ret, th3 = cv.threshold(img,80,255,cv.THRESH_BINARY)

  #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
  #                           cv.THRESH_BINARY,11,2)

#  k2 = np.ones((3,3), np.uint8)
  k2 = np.array([0,1,0,1,1,1,0,1,0], np.uint8).reshape((3,3))
  img_erosion = cv.dilate(th3, k2, iterations=1)
  img_d = cv.erode(img_erosion, k2, iterations=1)
#  img_d = cv.dilate(img_d, k2, iterations=1)

  img_d = cv.bitwise_not(img_d)
  return img_d 

def extractComponents(img):
  clean_im = clearImage(img)
#  cv.imshow("img1",clean_im);
#  cv.waitKey(0)
  num_labels, labels_im = cv.connectedComponents(clean_im)

  comps = [];
  for i in range(1,num_labels):
      c = extractComponent(labels_im, i)
      if(c['size'][0] > 5 and c['size'][1] > 5):
        comps.append(c)

  comps.sort(key=lambda c: c['topLeft'][0])

#  displayComponents(comps)

  return comps

def displayComponents(comps):
  for c in comps:
      print("Rectangle size: " + str(c['size']));
      cv.imshow(str(i) + '-th component',c['img'] );
      cv.waitKey(0)


def extractComponent(im, select_label):
 #   print("Size: " + str(im.shape))

    f =  np.full(im.shape, 255, "uint8")
    m_img = f *(im == select_label)
    r = m_img.view('uint8')

    x,y,w,h = cv.boundingRect(r)

    start = (x,y)
    end = (start[0] +w, start[1] + h)
    rc = r[start[1]:end[1], start[0]: end[0]]
    rc = cv.copyMakeBorder(rc, 0, max(35-h,0) , 0, max(30-w,0), cv.BORDER_CONSTANT, None, 0)
    rc = rc[0:35,0:30]
    result = {
        'img' : rc,
        'topLeft' : start,
        'bottomRight' : end,
        'size': (w,h)
    }

    return result;

def extractLabel(filename):
  f = os.path.basename(filename)
  name, ext = os.path.splitext(f)
  return name


def loadLabeledData(files):
  skipped = 0;
  letters = set()

  features = np.empty((0, 35 * 30))
  classes = np.empty(0)
  fileMask = np.zeros(len(files), np.bool)
  fileIndex = 0
  for file in files:
      im = cv.imread(file, 0);
    
  #    cv.imshow("img1",im);
  #    cv.waitKey(0)
      cs = extractComponents(im)
      label = extractLabel(file)

      for l in label:
          letters.add(l)

  #    print("Label: " + label)
  #    print(file + " has Cs: " + str(len(cs)))
      if(len(cs) == len(label)):
          fileMask[fileIndex] = True
          #print("Adding file with label: " + label)
          for i in range(0, len(cs)):
           # print("L:" + str(label[i]) + " size: " + str(cs[i]['img'].shape))
            cimg = np.ravel(cs[i]['img'])
            features = np.append(features, [cimg], axis=0)
            classes = np.append(classes, [label[i]], axis=0)
      else:
          skipped+=1
      fileIndex+=1

#  print("Skippede:" + str(skipped))
#  print("Letters size:" + str(len(letters)))
#  print("Letters: " + str(letters))

  return features, classes, fileMask
