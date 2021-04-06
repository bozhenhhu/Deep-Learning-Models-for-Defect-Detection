import argparse


def Hparams():
    parser = argparse.ArgumentParser()
    #network
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu', help='network activation')
    parser.add_argument('--l_weight', type=float, default=2., help='bce and dice weight')

    #whole project
    parser.add_argument('--width', type=int, default=256, help='image width for input')
    parser.add_argument('--height', type=int, default=256, help='image height for input')
    parser.add_argument('--Time', type=int, default=16)

    parser.add_argument('--seed', type=int, default=2020, help='seed for this program')
    parser.add_argument('--mode', type=str, default='train', help='train,test')
    parser.add_argument('--data_mode', type=str, default='3D', help='2D,3D')
    parser.add_argument('--ids', type=int, default=28)
    parser.add_argument('--sub_frame', type=str, default='first', help='first, last, last_mean, None')
    parser.add_argument('--mean_len', type=int, default=-10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--use_rotate', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='./mat2/plane/', help='path to load mat files')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--label_dir', type=str, default='./labels/')
    parser.add_argument('--heating_rate', type=int, default=4, help='4, or 5')
    parser.add_argument('--sample_rate', type=int, default=2)

    parser.add_argument('--model_mode', type=str, default='UNetpa', help='UNet++,UNet,UNet+pca, UNetpa')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=30)

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--draw_region', type=bool, default=False)

    #pca
    parser.add_argument('--pca_num', type=int, default=6, help='decide the num of images saved by PCA')
    args = parser.parse_args()

    return args
