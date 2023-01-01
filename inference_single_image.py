"""
单张图片推理 可视化代码

作者: 武家鹏
"""
import os
import argparse

import numpy as np
import cv2
from sklearn import svm
from sklearn.cluster import KMeans
from utils.dataset import DataLoader
from utils.cal_distance import cal_eculid_dist
import pickle

def main(opts):
    loader = DataLoader(data_root=opts.data_root)
    feature, label_name, label_dict, img = loader.get_features_and_labels_single(feature=opts.feature, mode='test', 
                                        resize_shape=opts.resize_shape, cell_size=opts.cell_size, gamma=opts.gamma)
    
    print('加载词袋...')
    codebooks = np.load('./weights/codebooks.npy')

    print('SVM推理...')
    feature_encoded = cal_eculid_dist(feature, codebooks).reshape(opts.cluster_num, )

    feature_encoded = np.asarray(feature_encoded).reshape(1, -1)

    # SVM_ = svm.LinearSVC()
    with open('./weights/svm.pickle', 'rb') as f:
        SVM_ = pickle.load(f)

    print('画图...')
    labels_pred = SVM_.predict(feature_encoded)

    # 根据字典对应关系找到类别的名称
    name_pred = ""
    for k in label_dict.keys():
        if label_dict[k] == labels_pred:
            name_pred = k
            break

    # 画图
    cv2.putText(img, f'TRUE CLASS: {label_name}', 
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3 ,color=(255, 0, 0), thickness=1)
    cv2.putText(img, f'PREDICT CLASS: {name_pred}', 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3 ,color=(255, 0, 0), thickness=1)
    cv2.imwrite(f'{np.random.randint(0, 2000)}.jpg', img)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/MSTAR2', help='数据集路径')
    parser.add_argument('--feature', type=str, default='hog', help='特征类别 hog 或 sift')
    parser.add_argument('--cluster_num', type=int, default=512, help='聚类个数')
    parser.add_argument('--resize_shape', type=int, default=128, help='resize的大小')
    parser.add_argument('--cell_size', type=int, default=16, help='HOG特征每个cell的大小')
    parser.add_argument('--gamma', type=float, default=1.2, help='gamma变换的系数')

    opts = parser.parse_args()

    main(opts)