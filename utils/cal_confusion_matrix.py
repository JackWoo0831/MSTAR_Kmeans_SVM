"""
计算混淆矩阵

作者: 武家鹏
"""
import numpy as np

def cal_confusion_matrix(labels_pred, labels_gt):
    """
    labels_pred: np.ndarray, 预测的类别
    labels_gt: np.ndarray, 真实的类别
    """
    classes = np.unique(labels_gt).shape[0]

    confusion_matrix = np.zeros(shape=(classes, classes))
    # 对于每一个GT 计数预测的类别
    for cls in range(classes):
        temp_idxs = labels_gt == cls  # 找到gt对应的该类别的索引
        temp_pred = labels_pred[temp_idxs]  # 找到模型都把cls预测成了什么

        for cls_pred in temp_pred:
            confusion_matrix[cls][cls_pred] += 1  # 计数 加1
        
    print('混淆矩阵:')
    print(confusion_matrix)

        