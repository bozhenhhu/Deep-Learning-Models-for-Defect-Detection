
from tensorflow.keras.callbacks import *
import copy
from sklearn.utils import shuffle

import matplotlib.image as mpimg
from augmentaton import *
from models import *
from utils import *


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return args.l_weight * tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)  # loss1


def read_img(path):  # read img of labels
    img = mpimg.imread(path)
    if len(img.shape) == 3:
        img = img[:, :, 0]
    return img


def one_data_process_plane3D(name0, path1, datas, labels):
    '''
    :param name0: mat name, eg: if one mat named 1.mat, then name0 = 1
    :param path1: path to load a specific mat data
    :param datas: list to save data
    :param labels: list to save labels
    :return:3D datas, labels
    '''
    label_path = args.label_dir + '{}_label.png'.format(name0)
    img = cv2.imread(label_path, 0)
    if img.shape != (args.height, args.width):
        img = cv2.resize(img, (args.height, args.width))
    img = img.astype(np.float32) / 255.

    data_struct = sio.loadmat(path1)
    data = data_struct['data']

    # subtract frame and sample
    t_len = data.shape[2]
    if args.sub_frame == 'first':
        sub = data[:, :, 0]
    elif args.sub_frame == 'last':
        sub = data[:, :, -1]
    elif args.sub_frame == 'last_mean':
        sub = data[:, :, args.mean_len:]
        sub = np.mean(sub, axis=2, keepdims=False)
    else:
        sub = np.zeros((data.shape[0], data.shape[1]))
    sub = sub.astype(np.float32) / 255.

    if t_len >= 220:
        data = data[:, :, -190:-30]  #data length:160
    elif t_len >= 190:
        data = data[:, :, 30:190]    #data length:160
    elif t_len >= 160:
        data = data[:, :, -160:]     #data length:160
    elif t_len >= 128:
        data = data[:, :, -128:]     #data length:128
    elif t_len >= 96:
        data = data[:, :, 30:94]     #data length:164
    elif t_len >= 64:
        data = data[:, :, 0:64]      #data length:64
    else:
        return datas, labels

    data = data.astype(np.float32) / 255.
    data = data - np.tile(sub[:, :, np.newaxis], (1, 1, data.shape[2]))

    data_ = np.zeros((args.Time, args.height, args.width))
    num = data.shape[2]//args.Time
    for k in range(num):
        d = list(np.arange(start=k, stop=data.shape[2], step=num))
        for i in range(len(d)):
            pre_frame = data[:, :, d[i]]
            data_[i, :, :] = cv2.resize(pre_frame, (args.height, args.width))
        datas.append(data_)
        labels.append(img)
    return datas, labels


def one_data_process_plane2D(name0, path1, datas, labels):
    '''
    :param name0: mat name, eg: if one mat named 1.mat, then name0 = 1
    :param path1: path to load a specific mat data
    :param datas: list to save data
    :param labels: list to save labels
    :return:2D datas, labels
    '''
    # label
    label_path = args.label_dir + '{}_label.png'.format(name0)
    img = cv2.imread(label_path, 0)
    if img.shape != (args.height, args.width):
        img = cv2.resize(img, (args.height, args.width))
    img = img.astype(np.float32) / 255.

    data_struct = sio.loadmat(path1)
    data = data_struct['data']

    # subtract frame
    t_len = data.shape[2]
    heating_num = t_len // args.heating_rate
    if args.sub_frame == 'first':
        sub = data[:, :, 0]
    elif args.sub_frame == 'last':
        sub = data[:, :, -1]
    elif args.sub_frame == 'last_mean':
        sub = data[:, :, args.mean_len:]
        sub = np.mean(sub, axis=2, keepdims=False)
    else:
        sub = np.zeros((data.shape[0], data.shape[1]))
    sub = sub.astype(np.float32) / 255.
    data = data[:, :, heating_num:min(t_len, heating_num+160)]
    data = data.astype(np.float32) / 255.
    data = data - np.tile(sub[:, :, np.newaxis], (1, 1, data.shape[2]))

    #sample and resize
    if args.use_rotate:
        solution = Solution()
        for i in range(0, data.shape[2], args.sample_rate):
            pre_frame = cv2.resize(data[:, :, i] , (args.height, args.width))
            datas.append(pre_frame)
            frame_copy = copy.deepcopy(pre_frame)

            frame_copy = solution.rotate(frame_copy)
            datas.append(frame_copy)

            labels.append(img)
            img_copy = copy.deepcopy(img)
            img_copy = solution.rotate(img_copy)
            labels.append(img_copy)
    else:
        for i in range(0, data.shape[2], args.sample_rate):
            pre_frame = data[:, :, i]
            if pre_frame.shape != (args.height, args.width):
                pre_frame = cv2.resize(pre_frame, (args.height, args.width))
            datas.append(pre_frame)
            labels.append(img)
    return datas, labels


def win_test_data_process(name0, path1, datas, num_slices, names):
    data_struct = sio.loadmat(path1)
    data = data_struct['data']

    # subtract frame
    t_len = data.shape[2]
    heating_num = t_len // args.heating_rate
    if args.sub_frame == 'first':
        sub = data[:, :, 0]
    elif args.sub_frame == 'last':
        sub = data[:, :, -1]
    elif args.sub_frame == 'last_mean':
        sub = data[:, :, args.mean_len:]
        sub = np.mean(sub, axis=2, keepdims=False)
    else:
        sub = np.zeros((data.shape[0], data.shape[1]))
    sub = sub.astype(np.float32) / 255.
    if args.data_mode == '2D':
        data = data[:, :, heating_num:min(t_len, heating_num + 160)]
        data = data.astype(np.float32) / 255.
        data = data - np.tile(sub[:, :, np.newaxis], (1, 1, data.shape[2]))
        data = data[:, :, heating_num:min(t_len, heating_num + 160)]
        for i in range(0, data.shape[2], 30):
            pre_frame = data[:, :, i]
            if pre_frame.shape != (args.height, args.width):
                pre_frame = cv2.resize(pre_frame, (args.height, args.width))
            datas.append(pre_frame)
            names.append(name0)
    elif args.data_mode == '3D':
        if t_len >= 220:
            data = data[:, :, -190:-30]  # data length:160
        elif t_len >= 190:
            data = data[:, :, 30:190]  # data length:160
        elif t_len >= 160:
            data = data[:, :, -160:]  # data length:160
        elif t_len >= 128:
            data = data[:, :, -128:]  # data length:128
        elif t_len >= 96:
            data = data[:, :, 30:94]  # data length:164
        elif t_len >= 64:
            data = data[:, :, 0:64]  # data length:64
        else:
            return datas, num_slices, names
        data = data.astype(np.float32) / 255.
        data = data - np.tile(sub[:, :, np.newaxis], (1, 1, data.shape[2]))

        data_ = np.zeros((args.Time, args.height, args.width))
        num = data.shape[2] // args.Time
        for k in range(1):
            d = list(np.arange(start=k, stop=data.shape[2], step=num))
            for i in range(len(d)):
                pre_frame = data[:, :, d[i]]
                data_[i, :, :] = cv2.resize(pre_frame, (args.height, args.width))
            datas.append(data_)
            names.append(name0)
    num_slices.append(len(datas))

    print('since added {}, the number of slices is {}'.format(names[-1], num_slices[-1]))
    return datas, num_slices, names


#data collected by 20200923 at Chengfei
Train_Name_Plane = ['0_20200615_4', '001g', '2_200615_1', '002g', '3_200615',
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


def get_files():
    filenames = os.listdir(args.data_path)
    datas = []
    labels = []
    for filename in filenames:
        name0 = os.path.splitext(filename)[0]
        name1 = os.path.splitext(filename)[1]
        if name1 == '.mat':
            path1 = args.data_path + filename
            if name0 in Train_Name_Plane:
                logging.info('train data name:{}'.format(name0))
                if args.data_mode == '2D':
                    datas, labels = one_data_process_plane2D(name0, path1, datas, labels)
                elif args.data_mode == '3D':
                    datas, labels = one_data_process_plane3D(name0, path1, datas, labels)

    datas = np.array(datas)
    labels = np.array(labels)
    datas = np.expand_dims(datas, axis=-1)
    labels[labels > 0.5] = 1
    labels[labels <= 0.5] = 0
    labels = np.expand_dims(labels, axis=-1)
    logging.info('train data shape:{}'.format(datas.shape))
    if args.shuffle:
        datas, labels = shuffle(datas, labels)
        return datas, labels
    else:
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
            if name0 not in Train_Name_Plane:
                print(name0)
                path1 = path + filename
                datas, num_slices, names = win_test_data_process(name0, path1, datas, num_slices,names)
    datas = np.array(datas)
    datas = np.expand_dims(datas, axis=-1)
    logging.info('test datas shape:{}'.format(datas.shape))
    return datas, num_slices, names



def main(ids):
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # set up the logger
    set_logger(os.path.join(args.log_path, 'train.log'))
    seeds(args.seed)
    logging.info(args)

    if args.mode == 'train':
        X_train, y_train = get_files()
        if args.model_mode == 'UNet++':
            model = Nest_Net()
        elif args.model_mode == 'UNet':
            model = UNet()
        elif args.model_mode == 'UNet+pca':
            model = UNet_plus3D_pca()
        elif args.model_mode == 'UNetpa':
            model = UNet_plus3D_st_pca()
        # model.load_weights('./huv25.h5',by_name=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=[bce_dice_loss],
                      metrics=["accuracy"])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,
                                      verbose=1, min_lr=1e-6)

        model_checkpoint = ModelCheckpoint('./v{}_{}_sub{}.h5'.format(ids, args.model_mode, args.sub_frame),
                                           monitor='val_loss',
                                           verbose=1, save_best_only=True)
        early_stoping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, mode='min',
                                      restore_best_weights=True)
        history = model.fit(X_train, y_train, batch_size=args.batchsize, epochs=args.epoch, verbose=1, validation_split=0.2,
                            callbacks=[reduce_lr, model_checkpoint, early_stoping])
        # history_df = pd.DataFrame(history.history)
        # history_df.to_csv('./history.csv', index=False)
    else:
        x_test, num_slices, names = win_test_files()
        model = tf.keras.models.load_model('./v{}_{}_sub{}.h5'.format(ids, args.model_mode, args.sub_frame),
                                           custom_objects={'bce_dice_loss': bce_dice_loss,
                                                            'dice_coef': dice_coef,
                                                            'dice_coef_loss': dice_coef_loss,
                                                            'width': args.width, 'height': args.height, 'Time':args.Time})
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
            if args.model_mode in ['UNet+pca', 'UNetpa'] and args.use_pca==False:
                y_array = y_predict[i, 10, :, :, 0]
                x_array = x_test[i, 10, :, :, 0]
            else:
                y_array = y_predict[i, :, :, 0]
                x_array = x_test[i, :, :, 0]
            x_array = np.array((x_array * 255.), dtype='uint8')
            y_array = np.array((y_array * 255.), dtype='uint8')
            xy = x_array + y_array
            xy = np.array((xy * 255.), dtype='uint8')
            xy = Image.fromarray(xy)
            xy.save('{}/v{}_xy_{}_{}.bmp'.format(save_path, ids, names[i], i), 'bmp')
            # plt.imsave('{}/hu{}_xy_{}_{}.bmp'.format(save_path, ids, names[i], i), xy, cmap='gray')

            if args.draw_region:
                temp_xy = regions(xy, y_array)
                temp_xy = Image.fromarray(temp_xy)
                temp_xy.save('{}/v{}_xy_{}_{}.bmp'.format(save_path, ids, names[i], i), 'bmp')


if __name__ == '__main__':
    main(ids=args.ids)




