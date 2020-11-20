
import tensorflow as tf
import numpy as np
import cv2
import os
import scipy.io as sio
from matplotlib import pyplot as plt
import matplotlib.image as mping
from matplotlib import cm
from research.hparams import Hparams

args = Hparams()


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
	return pca_


def get_files(read_path='./plane_0', save_path='./pca_imgs'):

	filenames = os.listdir(read_path)
	for filename in filenames:
		name0 = os.path.splitext(filename)[0]
		name1 = os.path.splitext(filename)[1]
		if name1 == '.mat':
			path1 = os.path.join(read_path, filename)
			data_struct = sio.loadmat(path1)
			data = data_struct['data']  # height, width, frame
			print(filename + ':' + str(data.shape))
			im = data[:, :, 0]
			# plt.imshow(im, 'gray')
			# plt.axis('off')
			# plt.show()
			h, w = im.shape
			new_data = []
			for i in range(data.shape[2]):
				temp = data[:, :, i].flatten()
				new_data.append(temp)
			new_data = tf.convert_to_tensor(new_data)
			pca_ = pca_tf(new_data, h, w).numpy()
			for i in range(args.pca_num):
				img = cv2.resize(pca_[i], (args.height, args.width), interpolation=cv2.INTER_AREA)
				mping.imsave('{}/{}_{}.png'.format(save_path, name0, i), img, cmap=cm.gray)
	print('pca finished')


get_files()





















