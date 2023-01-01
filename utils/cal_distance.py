"""
计算距离的函数集合

作者: 武家鹏
"""

import numpy as np  

def cal_eculid_dist(A, B):
    """
    计算A中每个行向量与B中每个行向量的欧氏距离

    A: shape(N, dim)
    B: shape(K, dim)

    return:
    shape(N, K)
    """
    if len(A.shape) == 1:  # 向量转换为矩阵
        A = A.reshape(1, -1)
    assert A.shape[1] == B.shape[1], f'A的维度是{A.shape}, B的维度是{B.shape}, 二者必须对应'

    result = np.zeros((A.shape[0], B.shape[0]))

    for i, row_A in enumerate(A):
        for j, row_B in enumerate(B):
            result[i, j] = np.dot((row_A - row_B).T, (row_A - row_B))
        
    return result

