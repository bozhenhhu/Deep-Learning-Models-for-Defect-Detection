import argparse


def Hparams():
    parser = argparse.ArgumentParser()
    #network
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--act', type=str, default='relu', help='network activation')
    #whole project
    parser.add_argument('--width', type=int, default=256, help='image width for input')
    parser.add_argument('--height', type=int, default=256, help='image height for input')
    parser.add_argument('--Time', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2020, help='seed for this program')
    parser.add_argument('--mode', type=int, default=0, help='mode 0 means training and mode 1 means testing')
    parser.add_argument('--ids', type=int, default=27)
    parser.add_argument('--sub_frame', type=bool, default=False)
    parser.add_argument('--use_rotate', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='./mat/plane/', help='path to load mat files')


    #pca
    parser.add_argument('--pca_num', type=int, default=6, help='decide the num of images saved by PCA')
    args = parser.parse_args()

    return args