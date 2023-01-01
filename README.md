# MSTAR_Kmeans_SVM
MSTAR dataset. Unoffical python implementation of Paper: [Bag-of-Visual-Words Based Feature Extraction for SAR Target Classification](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10420/1/Bag-of-visual-words-based-feature-extraction-for-SAR-target/10.1117/12.2281707.short)

MSTAR数据集的分类, 复现论文Bag-of-Visual-Words Based Feature Extraction for SAR Target Classification

## Performance: 

trian dataset acc: 90.45%  
test dataset acc: 85.37%

## Principles:

The train steps are as follow: 
1. gamma transform and extract HOG features 
2. use K-means cluster to build a codebook(Bag of word, that is some feature vectors can represent origin train data.) and save it  
3. use the Eculid distance between HOG feature and codebooks to encoder a new feature
4. train a Linear SVM model and save it
5. calculate the acc

The test steps are as follow: 
1. gamma transform and extract HOG features 
2. use the Eculid distance between HOG feature and codebooks to encoder a new feature
3. SVM prediction
4. calculate the acc

## How to use
### 1. Organizate the dataset

Download the dataset: 
> Link：https://pan.baidu.com/s/1Sa0AC9aERD36fpXGGYuI-w 
> Code：u4zg

unzip the `JPEG-e.7z`, and it should look something like this:

![png](https://github.com/JackWoo0831/MSTAR_Kmeans_SVM/blob/main/for_readme/1.png)

The first two folders are for the mixed type targets, the third and fourth folders are for the data of the T72 variant targets, and the last folder contains the T-72, BMP-2, BTR-70, SLICY targets. This repo focuses on the targets in the first two folders. According to the official recommendation, the images from the 17-degree imaging side view are used as the training set, and the images from the 15-degree imaging side view are used as the test set. It consists of 8 kinds of objectives, namely BTR_60, 2S1, BRDM_2, D7, SLICY, T62, ZIL131, ZSU_23_4.

**For convenience, all targets of 17-degree imaging are stored in the train folder, and all targets of 15-degree imaging are stored in the test folder**:

![png](https://github.com/JackWoo0831/MSTAR_Kmeans_SVM/blob/main/for_readme/2.png)


### 2. train
this repo relies on sklearn, opencv and numpy. you can run 

```shell
pip install -r requirements.txt
```

to install.

then run `train.py` directly or with some arguments:
```shell
python train.py 
```
or 

```shell
python train.py --data_root "D:/data/MSTAR" --cluster_num 64 --ratio 0.75
```

### 3. test

Same as train:  just run 
```shell
python test.py 
```
or

```shell
python test.py --data_root "D:/data/MSTAR" --cluster_num 64 --ratio 0.75
```
