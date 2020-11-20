import tensorflow as tf
from tensorflow.keras.layers import *


##attenstion
def SE(x, dim, ratio):
    squeeze = GlobalAveragePooling2D(name='squeeze')(x)
    se = Reshape((1, 1, dim), name='reshape')(squeeze)
    excitation = Dense(units=int(dim // ratio), name='dense_1', activation='relu', kernel_initializer='he_normal',
                       use_bias=False)(se)
    excitation = Dense(units=dim, name='dense_2', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(
        excitation)

    # excitation = Reshape((1,1,dim),name='reshape')(excitation)
    scale = multiply([x, excitation], name='multiply')
    return scale


def sc_SE(x, stage, dim, c_ratio, m, T, st_ratio):
    '''

    :param x: input
    :param stage: for name
    :param dim: num of filters
    :param c_ratio: the rate of reduce channels
    :param m: :the size of kernels
    :param T: T is real time
    :param st_ratio: the rate of reduce the m*n
    :return:
    '''
    se = GlobalAveragePooling3D(name=stage + 'squeeze')(x)
    se = Reshape((1, 1, 1, dim), name=stage + 'reshape_1')(se)
    se = Dense(units=int(dim // c_ratio), activation='relu', kernel_initializer='he_normal', name=stage + 'dense_1',
               use_bias=False)(se)
    se = Dense(units=int(dim), activation='sigmoid', kernel_initializer='he_normal', name=stage + 'dense_2',
               use_bias=False)(se)
    c_se = multiply([x, se], name=stage + 'multiply_1')

    ###3d spatial-time SE
    st_se = Conv3D(filters=1, kernel_size=1, strides=1, padding='valid', use_bias=False, name=stage + 'conv',
                   activation='relu', kernel_initializer='he_normal')(x)
    st_se = Flatten(name=stage + 'flatten')(st_se)
    st_se = Dense(units=max(2, int((m * m * T) // st_ratio)), activation='relu', name=stage + 'dense_3',
                  kernel_initializer='he_normal', use_bias=False)(st_se)
    st_se = Dense(units=int(m * m * T), activation='sigmoid', name=stage + 'dense_4',
                  kernel_initializer='he_normal', use_bias=False)(st_se)
    st_se = Reshape((T, m, m, 1), name=stage + 'reshape_2')(st_se)

    st_se = multiply([st_se, x], name=stage + 'multiply_2')

    result = Add(name=stage + 'add')([c_se, st_se])  # use sum
    return result


# 3D attention
def st_Attention(x, stage, t=1, k=3):
    st_se = Conv3D(filters=1, kernel_size=(t, k, k), padding='same', use_bias=False, name=stage + 'conv',
                   activation='sigmoid', kernel_initializer='he_normal')(x)
    scale = multiply([st_se, x], name=stage + 'multiply_2')
    return scale


def ECA(x, dim, k=3, gamma=2, b=1):
    '''
    paper:https://arxiv.org/abs/1910.03151?context=cs
    '''
    squeeze = GlobalAveragePooling2D(name='squeeze')(x)
    # t = int(abs((np.log2(dim)+b)/gamma))
    # k = t if t % 2 else t + 1 #3
    # print('se k:',k)
    attention = Reshape((dim, 1), name='reshape1')(squeeze)
    attention = Conv1D(filters=1, kernel_size=k, padding='same', activation='sigmoid', use_bias=False)(attention)
    attention = Permute((2, 1))(attention)
    attention = Reshape((1, 1, dim), name='reshape2')(attention)
    scale = multiply([x, attention], name='multiply')
    return scale
