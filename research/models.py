from tensorflow.keras.regularizers import l2
from hparams import Hparams
from attention import *
import tensorflow as tf


args = Hparams()


#network
def standard_unit3D(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv3D(nb_filter, (1, kernel_size, kernel_size), activation=args.act, name='conv' + stage + '_11',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Conv3D(nb_filter, (kernel_size, 1, 1), activation=args.act, name='conv' + stage + '_12',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Conv3D(nb_filter, (1, kernel_size, kernel_size), activation=args.act, name='conv' + stage + '_21',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Conv3D(nb_filter, (kernel_size, 1, 1), activation=args.act, name='conv' + stage + '_22',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    return x


def standard_unit2D(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=args.act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    # x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=args.act, name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    # x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
    return x


# this is tf version to implement pca, antirectifier and antirectifier_output_shape are used to bulid keras models' layers
def pca_tf(x, h, w):
    m, n = tf.compat.v1.to_float(x.get_shape()[0]), tf.compat.v1.to_int32(x.get_shape()[1])
    # assert not tf.assert_less(dim, n)

    mean = tf.reduce_mean(x, axis=0)
    x_new = x - mean

    cov = tf.matmul(x_new, x_new, transpose_b=True) / (m-1)
    e, v = tf.linalg.eigh(cov)  #e:eigenbalues/ v:eigenvetors
    #reduce dimension
    pca = tf.matmul(x_new, v, transpose_a=True)
    pca = tf.transpose(pca)
    pca = tf.reverse(pca, axis=[0])
    pca_ = []
    for i in range(args.pca_num):
        pca_.append(tf.reshape(pca[i, :], [h, w]))
    pca_ = tf.stack(pca_)
    pca_ = tf.reshape(pca_, [4, args.height, args.width])
    pca_ = tf.transpose(pca_, perm=[1, 2, 0])
    return pca_


def antirectifier(input):
    data = tf.squeeze(input, axis=[4])
    # batchsize,w,h = tf.to_int32(tf.shape(data)[0]),tf.to_int32(tf.shape(data)[1]),tf.to_int32(tf.shape(data)[2])
    outputs = []
    for i in range(args.batchsize):
        new_data = []
        for j in range(args.Time):
            new_data.append(tf.reshape(data[i, j, :, :], [1, args.height * args.width]))
        new_data = tf.stack(new_data)
        new_data = tf.reshape(new_data, [args.Time, args.height * args.width])
        tmp = pca_tf(new_data, args.height, args.width)

        outputs.append(tmp)
    outputs = tf.stack(outputs)
    return outputs


def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 5  # only valid for 3D tensors
    new_shape = list([shape[0], shape[2], shape[3], 4])
    return tuple(new_shape)


def UNet_plus3D_pca(num_class=1):
    nb_filter = [32, 64, 128, 256, 512]
    img_input = Input(shape=(args.Time, args.height, args.width, 1), name='main_input')
    conv1_1 = standard_unit3D(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit3D(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    up1_2 = Conv3D(nb_filter[0], 2, name='up12', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_1))
    conv1_2 = Concatenate(name='merge12', axis=4)([up1_2, conv1_1])
    conv1_2 = standard_unit3D(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit3D(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    up2_2 = Conv3D(nb_filter[1], 2, name='up22', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_1))
    conv2_2 = Concatenate(name='merge22', axis=4)([up2_2, conv2_1])
    conv2_2 = standard_unit3D(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3D(nb_filter[0], 2, name='up13', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_2))
    conv1_3 = Concatenate(name='merge13', axis=4)([up1_3, conv1_1, conv1_2])
    conv1_3 = standard_unit3D(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit3D(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    up3_2 = Conv3D(nb_filter[2], 2, name='up32', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv4_1))
    conv3_2 = Concatenate(name='merge32', axis=4)([up3_2, conv3_1])
    conv3_2 = standard_unit3D(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv3D(nb_filter[1], 2, name='up23', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_2))
    conv2_3 = Concatenate(name='merge23', axis=4)([up2_3, conv2_1, conv2_2])
    conv2_3 = standard_unit3D(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3D(nb_filter[0], 2, name='up14', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_3))
    conv1_4 = Concatenate(name='merge14', axis=4)([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = standard_unit3D(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit3D(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv3D(nb_filter[3], 2, name='up42', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv5_1))
    conv4_2 = Concatenate(name='merge42', axis=4)([up4_2, conv4_1])
    conv4_2 = standard_unit3D(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv3D(nb_filter[2], 2, name='up33', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv4_2))
    conv3_3 = Concatenate(name='merge33', axis=4)([up3_3, conv3_1, conv3_2])
    conv3_3 = standard_unit3D(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv3D(nb_filter[1], 2, name='up24', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_3))
    conv2_4 = Concatenate(name='merge24', axis=4)([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = standard_unit3D(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3D(nb_filter[0], 2, name='up15', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_4))
    conv1_5 = Concatenate(name='merge15', axis=4)([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = standard_unit3D(conv1_5, stage='15', nb_filter=nb_filter[0])

    ####################################
    if args.use_pca:
        if args.three_downsampling:
            conv1_4 = Lambda(lambda x: x[:, 10, :, :, :], name='sub_time')(conv1_4)
        else:
            conv1_4 = Lambda(lambda x: x[:, 10, :, :, :], name='sub_time')(conv1_5)

        pca = Lambda(function=antirectifier, output_shape=antirectifier_output_shape, name='lambda_pca')(img_input)
        pca = Activation('relu')(pca)
        pca = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(pca)
        #####
        pca = Concatenate(name='pca_merage', axis=3)([conv1_4, pca])
        conv1_4 = ECA(pca, dim=36)
        conv1_4 = Conv2D(36, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4),
                     name='pca_conv_1')(conv1_4)
        conv1_4 = Conv2D(36, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4),
                         name='pca_conv_2')(conv1_4)
        conv1_4_1 = Conv2D(2, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(num_class, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(conv1_4_1)
    else:
        conv1_4_1 = Conv3D(2, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1_4)
        nestnet_output_4 = Conv3D(num_class, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(conv1_4_1)

    model = tf.keras.Model(input=img_input, output=[nestnet_output_4])
    print(model.summary())
    return model


def UNet_plus3D_st_pca(num_class=1):
    nb_filter = [32, 64, 128, 256, 512]
    img_input = Input(shape=(args.Time, args.height , args.width, 1), name='main_input')
    conv1_1 = standard_unit3D(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit3D(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool2')(conv2_1)

    up1_2 = Conv3D(nb_filter[0], 2, name='up12', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_1))
    conv1_2 = Concatenate(name='merge12', axis=4)([up1_2, conv1_1])
    conv1_2 = standard_unit3D(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit3D(pool2, stage='31', nb_filter=nb_filter[2])
    conv3_1 = ECA(conv3_1, dim=nb_filter[2])
    conv3_1 = st_Attention(conv3_1, stage='31')
    pool3 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool3')(conv3_1)

    up2_2 = Conv3D(nb_filter[1], 2, name='up22', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_1))
    conv2_2 = Concatenate(name='merge22', axis=4)([up2_2, conv2_1])
    conv2_2 = standard_unit3D(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3D(nb_filter[0], 2, name='up13', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_2))
    conv1_3 = Concatenate(name='merge13', axis=4)([up1_3, conv1_1, conv1_2])
    conv1_3 = standard_unit3D(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit3D(pool3, stage='41', nb_filter=nb_filter[3])
    conv4_1 = ECA(conv4_1, dim=nb_filter[3])
    conv4_1 = st_Attention(conv4_1, stage='41')
    pool4 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='pool4')(conv4_1)

    up3_2 = Conv3D(nb_filter[2], 2, name='up32', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv4_1))
    conv3_2 = Concatenate(name='merge32', axis=4)([up3_2, conv3_1])
    conv3_2 = standard_unit3D(conv3_2, stage='32', nb_filter=nb_filter[2])
    conv3_2 = ECA(conv3_2, dim=nb_filter[2])
    conv3_2 = st_Attention(conv3_2, stage='32')

    up2_3 = Conv3D(nb_filter[1], 2, name='up23', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_2))
    conv2_3 = Concatenate(name='merge23', axis=4)([up2_3, conv2_1, conv2_2])
    conv2_3 = standard_unit3D(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3D(nb_filter[0], 2, name='up14', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_3))
    conv1_4 = Concatenate(name='merge14', axis=4)([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = standard_unit3D(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit3D(pool4, stage='51', nb_filter=nb_filter[4])
    conv5_1 = ECA(conv5_1, dim=nb_filter[4])
    conv5_1 = st_Attention(conv5_1, stage='51')

    up4_2 = Conv3D(nb_filter[3], 2, name='up42', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv5_1))
    conv4_2 = Concatenate(name='merge42', axis=4)([up4_2, conv4_1])
    conv4_2 = standard_unit3D(conv4_2, stage='42', nb_filter=nb_filter[3])
    conv4_2 = ECA(conv4_2, dim=nb_filter[3])
    conv4_2 = st_Attention(conv4_2, stage='42')

    up3_3 = Conv3D(nb_filter[2], 2, name='up33', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv4_2))
    conv3_3 = Concatenate(name='merge33', axis=4)([up3_3, conv3_1, conv3_2])
    conv3_3 = standard_unit3D(conv3_3, stage='33', nb_filter=nb_filter[2])
    conv3_3 = ECA(conv3_3, dim=nb_filter[2])
    conv3_3 = st_Attention(conv3_3, stage='33')

    up2_4 = Conv3D(nb_filter[1], 2, name='up24', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv3_3))
    conv2_4 = Concatenate(name='merge24', axis=4)([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = standard_unit3D(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3D(nb_filter[0], 2, name='up15', padding='same', activation='relu',
                   kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv2_4))
    conv1_5 = Concatenate(name='merge15', axis=4)([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = standard_unit3D(conv1_5, stage='15', nb_filter=nb_filter[0])

    ####################################
    if args.use_pca:
        if args.three_downsampling:
            conv1_4 = Lambda(lambda x: x[:, 10, :, :, :], name='sub_time')(conv1_4)
        else:
            conv1_4 = Lambda(lambda x: x[:, 10, :, :, :], name='sub_time')(conv1_5)

        pca = Lambda(function=antirectifier, output_shape=antirectifier_output_shape, name='lambda_pca')(img_input)
        pca = Activation('relu')(pca)
        pca = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(pca)
        #####
        pca = Concatenate(name='pca_merage', axis=3)([conv1_4, pca])
        conv1_4 = ECA(pca, dim=36)
        conv1_4 = Conv2D(36, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4),
                         name='pca_conv_1')(conv1_4)
        conv1_4 = Conv2D(36, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4),
                         name='pca_conv_2')(conv1_4)

        conv1_4_1 = Conv2D(2, 3, activation='relu', kernel_initializer='he_normal',padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(num_class, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(conv1_4_1)
    else:
        conv1_4_1 = Conv3D(2, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1_4)
        nestnet_output_4 = Conv3D(num_class, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                                  padding='same', kernel_regularizer=l2(1e-4))(conv1_4_1)

    model = tf.keras.Model(input=img_input, output=[nestnet_output_4])
    print(model.summary())
    return model


"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
def Nest_Net(img_rows=args.height, img_cols=args.width, color_type=1, num_class=1, deep_supervision=args.three_downsampling):
    nb_filter = [32, 64, 128, 256, 512]
    global bn_axis
    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    conv1_1 = standard_unit2D(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPool2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit2D(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit2D(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit2D(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit2D(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit2D(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit2D(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit2D(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit2D(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit2D(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit2D(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit2D(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit2D(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit2D(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit2D(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = tf.keras.Model(input=img_input, output=[nestnet_output_3])
    else:
        model = tf.keras.Model(inputs=img_input, outputs=[nestnet_output_4])
    print(model.summary())
    return model


def UNet():
    inputs = Input((None, None, 1))
    conv1 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    # conv1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    # conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))  ##fan conv
    merge1 = Concatenate(axis=3)([conv4, up1])
    conv6 = Conv2D(512,3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv6 = Conv2D(512,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge2 = Concatenate(axis=3)([conv3, up2])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge3 = Concatenate(axis=3)([conv2, up3])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge4 = Concatenate(axis=3)([conv1, up4])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)

    # conv10 = core.Reshape((nClasses, input_height * input_width))(conv10)
    model = tf.keras.Model(input=inputs,output=conv10)
    print(model.summary())
    return model

