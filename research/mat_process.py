
from tensorflow.keras.callbacks import *
import copy
from sklearn.utils import shuffle
import os
import warnings
import matplotlib.image as mpimg
from research.augmentaton import *
from research.models import *
from research.utils import *


def seeds(seed):
    warnings.filterwarnings("ignore")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seeds(args.seed)


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 2 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)  # loss1


def read_img(path):  # read img of labels
    img = mpimg.imread(path)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return img


def one_data_process_plane3D(name0, path1, datas, labels, sub_frame, use_rotate=False):
    '''
    :param name0: mat name, eg: if one mat named 1.mat, then name0 = 1
    :param path1: path to load a specific mat data
    :param datas: list to save data
    :param labels: list to save labels
    :param sub_frame:whether subtract the first frame or not
    :param use_rotate:whether use rotate or not
    :return:3D datas, labels
    '''
    label_path = './labels/{}_label.png'.format(name0)
    img = cv2.imread(label_path, 0)
    if img.shape != (args.height, args.width):
        img = cv2.resize(img, (args.height, args.width))
    img = img.astype(np.float32) / 255.
    data_struct = sio.loadmat(path1)
    data = data_struct['data']
    t_len = data.shape[2]
    if not sub_frame and not use_rotate:
        if t_len >= 220:
            data = data[:, :, -190:-30] #data length:160
        elif t_len >= 190:
            data = data[:, :, 30:190]   #data length:160
        elif t_len >= 160:
            data = data[:, :, -160:]    #data length:160
        elif t_len >= 128:
            data = data[:, :, -128:]    #data length:128
        elif t_len >= 96:
            data = data[:, :, 30:94]    #data length:164
        elif t_len >= 64:
            data = data[:, :, 0:64] #data length:64
        elif t_len >= 32:
            data = data[:, :, 0:32] #data length:32
        elif t_len >= 16:
            data = data[:, :, 0:16] #data length:16
        else:
            return datas, labels
        data_ = np.zeros((args.Time, args.height, args.width))
        num = data.shape[2]//args.Time
        for k in range(num):
            d = list(np.arange(start=k, stop=data.shape[2], step=num))
            for i in range(len(d)):
                pre_frame = data[:, :, d[i]]
                pre_frame = cv2.resize(pre_frame, (args.height, args.width))
                data_[i, :, :] = pre_frame.astype(np.float32) / 255.
            datas.append(data_)
            labels.append(img)
    return datas, labels


def one_data_process_plane2D(name0, path1, datas, labels, sub_frame, use_rotate=False):
    '''
    :param name0: mat name, eg: if one mat named 1.mat, then name0 = 1
    :param path1: path to load a specific mat data
    :param datas: list to save data
    :param labels: list to save labels
    :param sub_frame:whether subtract the first frame or not
    :param use_rotate:whether use rotate or not
    :return:2D datas, labels
    '''
    label_path = './labels/{}_label.png'.format(name0)
    img = cv2.imread(label_path, 0)
    if img.shape != (args.height, args.width):
        img = cv2.resize(img, (args.height, args.width))
    img = img.astype(np.float32) / 255.
    data_struct = sio.loadmat(path1)
    data = data_struct['data']
    first = data[:, :, 0]
    first = first.astype(np.float32) / 255.
    t_len = data.shape[2]
    heating_num = t_len//4
    data = data[:, :, heating_num:min(t_len, heating_num+160)]
    data = data.astype(np.float32) / 255.
    if sub_frame and use_rotate:
        solution = Solution()
        for i in range(0, data.shape[2], 2):
            pre_frame = cv2.resize((data[:, :, i] - first), (args.height, args.width))
            datas.append(pre_frame)
            frame_copy = copy.deepcopy(pre_frame)

            frame_copy = solution.rotate(frame_copy)
            datas.append(frame_copy)
            if img.shape != (args.height, args.width):
                img = cv2.resize(img, (args.height, args.width))
            labels.append(img)
            img_copy = copy.deepcopy(img)
            img_copy = solution.rotate(img_copy)
            labels.append(img_copy)
    elif not sub_frame and not use_rotate:
        for i in range(0, data.shape[2], 2):
            pre_frame = data[:, :, i]
            if pre_frame.shape != (args.height, args.width):
                pre_frame = cv2.resize(pre_frame, (args.height, args.width))
            datas.append(pre_frame)
            labels.append(img)
    return datas, labels


def win_test_data_process(name0, path1, datas, num_slices, names, sub_frame=False):
    data_struct = sio.loadmat(path1)
    data = data_struct['data']
    first = data[:, :, 0]
    first = first.astype(np.float32) / 255.
    t_len = data.shape[2]
    heating_num = t_len // 4
    data = data[:, :, heating_num:min(t_len, heating_num + 160)]
    data = data.astype(np.float32) / 255.
    if not sub_frame:
        for i in range(0, data.shape[2], 30):
            pre_frame = data[:, :, i]
            if pre_frame.shape != (args.height, args.width):
                pre_frame = cv2.resize(pre_frame, (args.height, args.width))
            datas.append(pre_frame)
            names.append(name0)

    num_slices.append(len(datas))

    print('since added {}, the number of slices is {}'.format(names[-1], num_slices[-1]))
    return datas, num_slices, names


#data collected by 20200923 at Chengfei
train_name_plane = ['0_20200615_4', '001g', '2_200615_1', '002g', '3_200615',
                    '005g', '046g', '051g', '166g',
                    '0602(1)', '0602(4)', '0602(5)', '0602(2)',
                    '20200618_0110g','20200618_0111g', '20200618_0112g', '20200618_0115g',
                    '20200618_0116g', '20200618_0117g', '20200618_0118g','20200618_0120g', '20200618_0121g',
                    '20200618_0122g', '20200618_0123g', '20200618_0124g', '20200618_0127g', '20200618_0128g',
                    '20200618_0129g',
                    '20200722_0031g', '20200723_0014g', '20200723_0015g', '20200723_0037g',
                    '20200723_0038g', '20200723_0040g', '20200723_0049g','20200812_0001g',
                    '20200812_0002g', '20200812_0003g', '20200812_0004g', '20200812_0005g', '20200812_0006g',
                    '20200812_0011g', '20200812_0012g', '20200813_0001g', '20200813_0002g',
                    '20200923_0003g', '20200923_0004g', '20200923_0005g', '20200923_0006g',
                    '20200923_0008g', '20200923_0009g', '20200923_0010g', '20200923_0012g',
                    '20200923_0014g', '20200923_0015g', '20200923_0016g', '20200923_00017g',
                    '20200923_0018g', '20200923_0021g']


def get_files(Shuffle=True, path='./mat/plane/', sub_frame=False):
    '''
    :param Shuffle: whether shuffle data
    :param path: path to load mat files
    :param sub_frame:whether subtract the first frame
    :return:training data and labels
    '''
    filenames = os.listdir(path)
    datas = []
    labels = []
    for filename in filenames:
        name0 = os.path.splitext(filename)[0]
        name1 = os.path.splitext(filename)[1]
        if name1 == '.mat':
            path1 = path + filename
            if name0 in train_name_plane:
                print(name0)
                datas, labels = one_data_process_plane2D(name0, path1, datas, labels, sub_frame)

    datas = np.array(datas)
    labels = np.array(labels)
    datas = np.expand_dims(datas, axis=-1)
    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0
    labels = np.expand_dims(labels, axis=-1)
    assert len(datas.shape) == 4
    assert len(labels.shape) == 4
    if Shuffle:
        print('train data shape:{}'.format(datas.shape))
        datas, labels = shuffle(datas, labels)

        return datas, labels
    else:
        print(datas.shape)
        return datas, labels


def win_test_files(path='./mat/plane/'):
    '''
    :param path: path to load test mat data
    :return: test datas, a list, test data mat names
    '''
    filenames = os.listdir(path)
    datas = []
    num_slices = []
    names = []
    for filename in filenames:
        name0 = os.path.splitext(filename)[0]
        name1 = os.path.splitext(filename)[1]
        if name1 == '.mat':
            if name0 not in train_name_plane:
                print(name0)
                path1 = path + filename
                datas, num_slices, names = win_test_data_process(name0, path1, datas, num_slices,
                                                                        names)
    datas = np.array(datas)
    datas = np.reshape(datas, (-1, args.height, args.width))
    datas = np.expand_dims(datas, -1)
    print('datas shape', datas.shape)
    return datas, num_slices, names


def main(ids):
    args.sub_frame = False
    if args.mode == 0:
        X_train, y_train = get_files(Shuffle=True, path=args.data_path, sub_frame=args.sub_frame)

        model = Nest_Net()
        # model.load_weights('./huv25.h5',by_name=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.binary_crossentropy,
                      metrics=["accuracy"])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,
                                      verbose=1, min_lr=1e-6)

        model_checkpoint = ModelCheckpoint('./huv{}.h5'.format(ids), monitor='val_loss',
                                           verbose=1, save_best_only=True)
        early_stoping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, mode='min',
                                      restore_best_weights=True)
        history = model.fit(X_train, y_train, batch_size=16, epochs=30, verbose=1, validation_split=0.2,
                            callbacks=[reduce_lr, model_checkpoint, early_stoping])
        # history_df = pd.DataFrame(history.history)
        # history_df.to_csv('../portable2/history.csv', index=False)
    else:
        x_test, num_slices, names = win_test_files()
        model = tf.keras.models.load_model('./huv{}.h5'.format(ids), custom_objects={'bce_dice_loss': bce_dice_loss,
                                                                                  'dice_coef': dice_coef,
                                                                                  'dice_coef_loss': dice_coef_loss,
                                                                                  'width': args.width, 'height': args.height})
        y_predict = model.predict(x=x_test, batch_size=2)

        #########
        save_path = './DefectNet/Result{}/'.format(ids)
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
            x_array = x_test[i, :, :, 0]
            x_array = np.array((x_array * 255.), dtype='uint8')
            y_array = np.array((y_array * 255.), dtype='uint8')
            xy = x_array + y_array
            xy = np.array((xy * 255.), dtype='uint8')
            xy = Image.fromarray(xy)
            xy.save('{}/hu{}_xy_{}_{}_1.bmp'.format(save_path, ids, names[i], i), 'bmp')

            # temp_xy = regions(xy, y_array)
            # temp_xy = Image.fromarray(temp_xy)
            # temp_xy.save('{}/hu{}_xy_{}_{}.bmp'.format(save_path, ids, names[i], i), 'bmp')


args.mode = 1
main(ids=args.ids)




