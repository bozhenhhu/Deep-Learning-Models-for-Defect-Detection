'''
The model mainly based on 3D UNet++ is designed for flat type data on win10 system
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #-1 for cpu and 0 for gpu
import sys
####you need change this path which depends on your own tensorflow directory
sys.path.extend(['D:\\Anaconda3\\envs\\tfenv\\Lib\\site-packages'])
sys.path.extend(['D:\\Anaconda3\\Lib\\site-packages'])


from keras.callbacks import *
import selectivesearch as ss
import keras

import numpy as np
import cv2
from PIL import Image
import csv


External_Parameter = sys.argv
base_path = External_Parameter[1]
log_path = base_path+'\\DefectNet\\test.log'
fp = open(log_path ,'w')
stderr = sys.stderr
stdout = sys.stdout
sys.stderr = fp
sys.stdout = fp
print('first external param:{}\nsecond external param:{}'.format(External_Parameter[0], base_path))
smooth = 1.
dropout_rate = 0.5
act = "relu"
width = 192
height = 192
Time = 16


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 2 * keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)  # loss1


def regions(temp_xy, y_array):
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


def data_process(data, t_len):
    if t_len >= 220:
        data = data[:, :, -190:-30]
    elif t_len >= 190:
        data = data[:, :, 30:190]
    elif t_len >= 160:
        data = data[:, :, -160:]
    elif t_len >= 128:
        data = data[:, :, -128:]
    elif t_len >= 96:
        data = data[:, :, 30:94]
    elif t_len >= 64:
        data = data[:, :, 0:64]
    elif t_len >= 32:
        data = data[:, :, 0:32]
    elif t_len >=16:
        data = data[:, :, 0:16]
    else:
        logging.info('the frames are too short to be processed')
        sys.exit(0)

    data_ = np.zeros((Time, width, height))
    datas = []
    num = data.shape[2] // Time
    for k in range(1):
        d = list(np.arange(start=k, stop=data.shape[2], step=num))
        for i in range(len(d)):
            pre_frame = data[:, :, d[i]]
            pre_frame = pre_frame.astype(np.float32) / 255.
            data_[i, :, :] = cv2.resize(pre_frame, (height, width))
        datas.append(data_)
    datas = np.array(datas)
    datas = np.expand_dims(datas, axis=-1)
    print('test datas shape: ', datas.shape)
    assert len(datas.shape) == 5
    return datas


def read_data(csv_path):
    '''
    read csv data from disk
    :return: data ndarray format
    '''
    with open(csv_path) as csvFile:
        readcsv = csv.reader(csvFile)
        rows = [list(map(lambda x:int(x),row)) for row in readcsv]
        rows = np.array(rows)
    print('the shape of csv data: {}'.format(rows.shape))
    t_length = rows.shape[1]
    assert len(rows) == 307200
    assert len(rows.shape) == 2
    rows = rows.reshape(480, 640, t_length)
    data = data_process(rows, t_length)
    print('python program has read and processed csv data successfully')
    return data


def main(ids):
    df = read_data(base_path + '\\DefectNet\\defectnetinput.csv')
    custom_objects = {'bce_dice_loss': bce_dice_loss, 'dice_coef': dice_coef,
                      'dice_coef_loss': dice_coef_loss,
                      'width': width, 'height': height, 'Time': Time}
    model = keras.models.load_model(base_path+'/DefectNet/huv{}.h5'.format(ids), custom_objects=custom_objects)
    print('load model successfully')
    y_predict = model.predict(x=df, batch_size=1)
    print('predict successfully')

    save_path = base_path + '\\DefectNet\\Result\\'
    print('start removing files in {} if it existed'.format(save_path))
    if os.path.exists(save_path):
        files = os.listdir(save_path)
        for file in files:
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(save_path)

    for i in range(y_predict.shape[0]):
        y_array = y_predict[i, :, :, 0]
        x_array = df[i, 10, :, :, 0]
        x_array = np.array((x_array * 255.), dtype='uint8')
        y_array = np.array((y_array * 255.), dtype='uint8')
        xy = x_array + y_array
        xy = np.array((xy * 255.), dtype='uint8')
        xy = Image.fromarray(xy)

        temp_xy = regions(xy, y_array)
        temp_xy = Image.fromarray(temp_xy)
        temp_xy.save('{}/hu{}_xy_{}.bmp'.format(save_path, ids, i), 'bmp')


main(ids=26)
fp.close()
sys.stderr = stderr
sys.stdout = stdout
