"""
训练代码

基本思路:
1. 提取图像特征
2. k-means聚类
3. 对于一个样本 与聚类中心点结合 进一步编码特征

作者: 武家鹏
"""

import os
import argparse

import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from utils.dataset import DataLoader
from utils.cal_distance import cal_eculid_dist
import pickle

def main(opts):
    
    # 1. 加载数据集并得到特征与标签
    print('加载数据集...')
    loader = DataLoader(data_root=opts.data_root)
    features_trian, labels_train = loader.get_features_and_labels(feature=opts.feature, mode='train', 
                                        resize_shape=opts.resize_shape, cell_size=opts.cell_size, gamma=opts.gamma)

    # 2. k-means聚类获得codebooks
    print('聚类...')
    k_means_cluster = KMeans(n_clusters=opts.cluster_num)
    k_means_fitter = k_means_cluster.fit(features_trian)

    codebooks = k_means_fitter.cluster_centers_  # 聚类中心即为所谓的bag of words  shape: (cluster_num, n_dim)
    # 保存结果
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    np.save('./weights/codebooks.npy', codebooks)
    # 3. SVM训练
    # 对每个样本 计算与每个聚类中心的欧氏距离 编码为新特征
    
    print('训练SVM...')
    features_trian_encoded = []
    for feature in features_trian:
        features_trian_encoded.append(cal_eculid_dist(feature, codebooks).reshape(opts.cluster_num, ))

    features_trian_encoded = np.asarray(features_trian_encoded)
    SVM_ = svm.LinearSVC()
    SVM_.fit(features_trian_encoded, labels_train)
    # 保存结果
    with open('./weights/svm.pickle', 'wb') as f:
        pickle.dump(SVM_, f)

    # 4. 评估训练集准确率
    print('评估...')
    labels_pred = SVM_.predict(features_trian_encoded)
    acc = sum( labels_pred == labels_train) / len(labels_train)
    print('训练集准确率为{}%'.format(acc * 100))


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
