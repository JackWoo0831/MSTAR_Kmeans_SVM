"""
测试代码

基本思路:
1. 读取存储的聚类中心点(codebooks)和SVM模型权重
2. 推理
3. 评估准确率

作者: 武家鹏
"""

import os
import argparse

import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from utils.dataset import DataLoader
from utils.cal_distance import cal_eculid_dist
from utils.cal_confusion_matrix import cal_confusion_matrix
import pickle

def main(opts):
    
    # 1. 加载数据集并得到特征与标签
    print('加载数据集...')
    loader = DataLoader(data_root=opts.data_root)
    features_test, labels_test = loader.get_features_and_labels(feature=opts.feature, mode='test', 
                                        resize_shape=opts.resize_shape, cell_size=opts.cell_size, gamma=opts.gamma)

    # 2. 加载codebook
    print('加载词袋...')

    codebooks = np.load('./weights/codebooks.npy')  # 聚类中心即为所谓的bag of words  shape: (cluster_num, n_dim)

    # 3. SVM推理
    # 对每个样本 计算与每个聚类中心的欧氏距离 编码为新特征
    
    print('SVM推理...')
    features_test_encoded = []
    for feature in features_test:
        features_test_encoded.append(cal_eculid_dist(feature, codebooks).reshape(opts.cluster_num, ))

    features_test_encoded = np.asarray(features_test_encoded)
    # SVM_ = svm.LinearSVC()
    with open('./weights/svm.pickle', 'rb') as f:
        SVM_ = pickle.load(f)

    # 4. 评估训练集准确率
    print('评估...')
    labels_pred = SVM_.predict(features_test_encoded)
    acc = sum( labels_pred == labels_test) / len(labels_test)
    print('测试集准确率为{}%'.format(acc * 100))

    cal_confusion_matrix(labels_pred, labels_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/MSTAR2', help='数据集路径')
    parser.add_argument('--feature', type=str, default='hog', help='特征类别 hog 或 sift')
    parser.add_argument('--cluster_num', type=int, default=256, help='聚类个数')
    parser.add_argument('--resize_shape', type=int, default=128, help='resize的大小')
    parser.add_argument('--cell_size', type=int, default=16, help='HOG特征每个cell的大小')
    parser.add_argument('--gamma', type=float, default=1.2, help='gamma变换的系数')

    opts = parser.parse_args()

    main(opts)