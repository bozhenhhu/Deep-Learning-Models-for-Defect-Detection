import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import scipy.io as sio


def loadData(img_path):
    data = []
    img = image.open(img_path)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            pixel = img.getpixel((i,j))
            # x, y, z = img.getpixel((i, j))
            data.append([pixel / 256.0, i / float(m), j / float(n)])
    return np.mat(data), m, n


def kmeans(imgData, row, col, cluster_num=2):

    label = KMeans(n_clusters=cluster_num).fit_predict(imgData)

    label = label.reshape([row, col])
    label = np.transpose(label,(1,0))

    plt.imshow(label,cmap='gray')
    plt.axis('off')
    plt.show()


path = ''
imgData, row, col = loadData(path)
kmeans(imgData, row, col)