import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    copy = []
    for seed in seeds:
        seedList.append(seed)
        copy.append(seed)
    label = 1
    connects = selectConnects(p)

    while (len(seedList) > 0):#the condition for finishing the circulation
        print('len(seedList):',len(seedList))
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:  #the law fo growing
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


def on_mouse(event, x, y, flag, params):
    clicks = []
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + 'Point' + '('+str(x) + ', ' + str(y)+')')
        clicks.append((y, x))
    return clicks


img_path = ''
img = cv2.imread(img_path,0)
# cv2.namedWindow('image')
# clicks = cv2.setMouseCallback('image',on_mouse)
# cv2.imshow('image',img)
# cv2.waitKey(30000)
# cv2.destroyAllWindows()
seeds = [Point(67, 51), Point(63, 102), Point(65, 165),Point(59, 216),Point(54, 273), \
         Point(138, 50),Point(136, 115),Point(131, 174),Point(129, 277)]
binaryImg = regionGrow(img, seeds, 0.5)

cv2.imshow('',binaryImg)
cv2.waitKey(0)
