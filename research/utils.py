import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import os
import warnings
import random
import tensorflow as tf

def seeds(seed):
    warnings.filterwarnings("ignore")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)


def on_mouse(event, x,y, flags , params):
    clicks = []
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + 'Point' + '('+str(x) + ', ' + str(y)+')')
        clicks.append((y, x))
    return clicks


def get_position(img_path):
    '''
    :param img_path: path to read an img
    :return: through click on the img by mouse, we can get (x,y) coordinates on this img
    '''
    img = cv2.imread(img_path, 0)
    cv2.namedWindow('image')
    clicks = cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image',img)
    cv2.waitKey(60000)
    cv2.destroyAllWindows()


def plot_curve(mat_path):
    '''
    :param mat_path: path to read a mat file
    :return: draw Temperature Change Curve for some points that you listed in this function
    '''
    data_struct = sio.loadmat(mat_path)  #if 195t.mat
    data = data_struct['data']
    x = [x for x in range(data.shape[2])]
    y1 = data[197, 151, x]
    y2 = data[209, 149, x]
    y3 = data[212, 151, x]
    y4 = data[251, 149, x]
    y5 = data[251, 180, x]
    plt.figure()
    plt.plot(x, y1, color='dodgerblue', label='Line1')
    plt.plot(x, y2, color='orangered', label='Line2')
    plt.plot(x, y3, color='orange', label='Line3')
    plt.plot(x, y4, color='mediumorchid', label='Line4')
    plt.plot(x, y5, color='limegreen', label='Line5')

    plt.ylim((24, 36))
    plt.xlim((0, 200))
    y_ticks = np.linspace(24, 36, num=6)
    x_ticks = np.linspace(0, 200, num=5)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Centigrade Temperature(℃)')
    plt.title('Temperature Change Curve')
    plt.show()
    plt.savefig('./Line.png')
    return


def plot_3Ddata(data):
    #plot one mat data depended on values on time  and locations
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    X = [X for X in range(data.shape[0])]
    Y = [Y for Y in range(data.shape[1])]
    X, Y = np.meshgrid(X, Y)

    Z = data[X, Y]
    ax.plot_surface(X, Y, Z,cmap = 'rainbow')
    plt.show()


def jpg2png():
    import glob
    from os.path import splitext
    files = glob.glob("./*.jpg")  #path
    for jpg in files:
        im = Image.open(jpg)
        png = splitext(jpg)[0] + "." + "png"
        im.save(png)
        # print(jpg)


#rotate an image of 90 degree
class Solution:
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(i + 1):
                self.swap(matrix, i, j, j, i)
        for i in range(n // 2):
            for j in range(m):
                self.swap(matrix, j, i, j, n - 1 - i)
        return matrix

    def swap(self, matrix, i1, j1, i2, j2):
        temp = matrix[i1][j1]
        matrix[i1][j1] = matrix[i2][j2]
        matrix[i2][j2] = temp


#selective search, draw rectangles
def regions(temp_xy, y_array):
    import selectivesearch as ss
    temp_xy = np.expand_dims(temp_xy, axis=-1)
    temp_xy = np.tile(temp_xy, [1, 1, 3])
    y_array = np.expand_dims(y_array, axis=-1)
    y_array = np.tile(y_array, [1, 1, 3])

    img_lbl, regions = ss.selective_search(y_array, scale=2500, sigma=0.9, min_size=10)

    eps = 1e-7

    candidates = set()
    for r in regions:
        if r['rect'] in candidates:  # remove duplicate
            continue
        if r['size'] > 6000:  # remove the biggest rect
            continue
        x, y, w, h = r['rect']
        s = r['size']
        if w / (h + eps) > 1.5 or h / (w + eps) > 1.5:
            continue
        candidates.add((x, y, w, h, s))
    candidates = list(candidates)

    def select_candidates(candidates):
        a = []
        for i in range(len(candidates)):
            if len(a) == 0:
                a.append(list(candidates[i]))
            else:
                for j in range(len(a)):
                    if abs(a[j][0] - candidates[i][0]) < 10 and abs(a[j][1] - candidates[i][1]) < 10:
                        if a[j][0] - candidates[i][0] > 0 and a[j][1] - candidates[i][1] > 0:
                            break
                        else:
                            a[j][0] = candidates[i][0]
                            a[j][1] = candidates[i][1]
                            break
                    else:
                        if j == len(a) - 1:
                            a.append(list(candidates[i]))
                            break
        return a

    candidates = select_candidates(candidates)
    for r in candidates:
        x, y, w, h, s = r

        # cv2.rectangle(temp_xy, (x, y), (x + h, y + w), (255, 0, 0), 1)
        cv2.putText(temp_xy, 's' + str(s), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 0, 0), 1)
    return temp_xy
