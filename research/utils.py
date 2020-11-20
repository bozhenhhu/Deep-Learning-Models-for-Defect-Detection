import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

