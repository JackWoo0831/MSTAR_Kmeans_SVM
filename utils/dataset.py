"""
定义数据集加载类

读取数据

作者: 武家鹏
"""
import os 
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_root) -> None:
        """
        data_root: str, 数据路径
        """
        self.data_root = data_root
        self.categories = sorted(os.listdir(os.path.join(data_root, 'train')))
        self.label_dict = {self.categories[i]: i for i in range(len(self.categories))}

        print(f'待分类的类别与对应序号: {self.label_dict}')

    
    def get_features_and_labels(self, feature='hog', mode='train', 
            resize_shape=128, cell_size=32, num_cell_per_block=2, shuffle=True, gamma=1):
        """
        feature: str, 特征类型
        mode: str, 读取训练集还是测试集
        resize_shape: int, 要将图片resize的尺寸, 图片被resize成等长宽.
        cell_size: int, 划分成cell的个数 HOG特征
        num_cell_per_block: int, 每个cell中block数目 HOG特征
        shuffle: bool, 是否打乱数据集

        return:
        features, np.ndarray, shape: (N_samples, feature_dim)
        labels, np.ndarray, shape: (N_samples, )
        """

        if not feature == 'hog':
            raise NotImplementedError  # SIFT 暂未实现

        features = []
        labels = []

        current_path = os.path.join(self.data_root, mode)  # 进入trian或test文件夹下
        # 定义HOG算子
        win_size, block_size, block_stride, cell_size = self.set_hog_params([resize_shape, resize_shape], 
                                                            (cell_size, cell_size), (num_cell_per_block, num_cell_per_block))
        descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)

        # 读取文件夹下每个子目录的图片 预处理后提取特征

        table = [((i / 255) ** gamma * 255) for i in range(256)]  # 先计算gamma查找表 防止重复计算
        table = np.array(table).astype('uint8')

        for cat in self.categories:
            imgs = os.listdir(os.path.join(current_path, cat))

            for img_name in imgs:
                img = cv2.imread(os.path.join(current_path, cat, img_name), cv2.IMREAD_GRAYSCALE)
                img = self.preprocess(img, resize_shape=resize_shape, gamma_table=table)
                features.append(descriptor.compute(img))
                labels.append(self.label_dict[cat])

        return np.asarray(features), np.asarray(labels)


    def get_features_and_labels_single(self, feature='hog', mode='train', 
            resize_shape=128, cell_size=32, num_cell_per_block=2, shuffle=True, gamma=1):
        """
        获取单张图片的特征和标签
        参数与get_features_and_labels相同
        """
        data_root = os.path.join(self.data_root, 'test')
        cls_list = os.listdir(data_root)
        cls = cls_list[np.random.randint(0, len(cls_list))]  # 随机抽取一个类别

        imgs_path = os.listdir(os.path.join(data_root, cls))
        img = imgs_path[np.random.randint(0, len(imgs_path))]  # 随机抽取一个图片
        img = cv2.imread(os.path.join(data_root, cls, img), cv2.IMREAD_GRAYSCALE)

        # 定义HOG算子
        win_size, block_size, block_stride, cell_size = self.set_hog_params([resize_shape, resize_shape], 
                                                            (cell_size, cell_size), (num_cell_per_block, num_cell_per_block))
        descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)

        table = [((i / 255) ** gamma * 255) for i in range(256)]  # 先计算gamma查找表 防止重复计算
        table = np.array(table).astype('uint8')

        img = self.preprocess(img, resize_shape=resize_shape, gamma_table=table)

        return descriptor.compute(img), cls, self.label_dict, img

    
    def preprocess(self, img, resize_shape, gamma_table):
        """
        预处理图像 裁剪+gamma变换

        img: np.ndarray 读取的图像
        """

        # 如果图片大小比(resize_shape, resize_shape)大, 则以中心裁剪 否则填充
        img_shape = img.shape
        if img.size >= resize_shape ** 2:  # 如果需要裁剪
            center_x, center_y = img_shape[0] // 2, img_shape[1] // 2  # 计算中心点
            # 计算裁剪的边界
            tl, tr = center_x - resize_shape // 2, center_x + resize_shape // 2
            bl, br = center_y - resize_shape // 2, center_y + resize_shape // 2
            # 进行裁剪
            img = img[tl: tr, bl: br]
        else:  # 如果需要填充
            # 计算上下左右分别需要填充多少
            padding_sum = resize_shape - img_shape[0]
            left, right = padding_sum // 2, padding_sum - padding_sum // 2
            padding_sum = resize_shape - img_shape[1]
            top, bottom = padding_sum // 2, padding_sum - padding_sum // 2

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        assert img.size == resize_shape ** 2

        # gamma变换        
        img = cv2.LUT(img, gamma_table)

        return img


    def norm_images(self, img):
        """
        归一化图像

        img: np.ndarray 读取的图像
        """
        img_ = img
        img_ = img_ / img_.max()
        img_ *= 255
        return img_.astype(np.uint8)
        
            
    def set_hog_params(self, img_shape, cell_size, num_cell_per_block):
        """
        设置HOG描述子参数

        img_shape: tuple or list, 图像高宽
        cell_size: tuple or list, 每个cell的高宽
        num_cell_per_block:, int, 每个block中有多少cell
        """
        block_size = (num_cell_per_block[0] * cell_size[0], num_cell_per_block[1] * cell_size[1])
        x_cells = img_shape[1] // cell_size[0]
        y_cells = img_shape[0] // cell_size[1]
        h_stride = 1
        v_stride = 1
        block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
        win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])

        return win_size, block_size, block_stride, cell_size
