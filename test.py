import numpy as np
import cv2 as cv
import utils as cp



def dopic(img):
    margin = 2
    img = img[margin:-margin,margin:-margin]
    ret, th3 = cv.threshold(img,127,255,cv.THRESH_BINARY)

    k2 = np.ones((3,3), np.uint8)

    img_erosion = cv.dilate(th3, k2, iterations=1)
    img_d = cv.erode(img_erosion, k2, iterations=1)

    img_d = cv.bitwise_not(img_d)

    #img_e2 = cv.dilate(img_d, kernel2, iterations=1)
    #img_d2 = cv.erode(img_e2, kernel2, iterations=1)




    cv.imshow('image2', th3);
    cv.waitKey(0)
    cv.imshow('image3', img_d);
    cv.waitKey(0)



    num_labels, labels_im = cv.connectedComponents(img_d)

    imshow_components(labels_im)


    print("Size orig: " + str(img_d.shape))

    comps = [];
    for i in range(1,num_labels):
        c = extractComponent(labels_im, i)
        comps.append(c)

    comps.sort(key=lambda c: c['topLeft'][0])

    for c in comps:
        print("Rectangle: " + str(c['topLeft']));
        cv.imshow(str(i) + '-th component',c['img'] );
        cv.waitKey(0)




    cv.destroyAllWindows()

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv.imshow('labeled.png', labeled_img)
    cv.waitKey()


img = cv.imread('labeled/hb8pa.jpg', 0)

i1 = cp.clearImage(img);

cv.imshow('image3', i1);
cv.waitKey(0)

num_labels, labels_im = cv.connectedComponents(i1)
imshow_components(labels_im)
