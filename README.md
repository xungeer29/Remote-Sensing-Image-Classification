# Remote-Sensing-Image-Classification
## 数据集
一个纯净的、没有噪声的遥感图像数据集，共21类，每类100张图像，可以用于分类任务的入门练手

在本次的项目中，将数据集按照 8:2 随机划分为训练集和验证集

```
链接: https://pan.baidu.com/s/1Avcih8rARD2LoBp5S4n2ww 
提取码: hp5f
```
## ENVS
* python==2.7
* pytorch==0.4.1

## 目录结构
```
remote_sensing_image_classification/
▾ data/
    label_list.txt
    train.txt
    valid.txt
▾ dataset/
    __init__.py
    create_img_list.py
    dataset.py
▾ figs/
    acc.eps
    acc.jpg
    confusion_matrix.jpg
▾ log/
    log.txt
▾ metrics/
    __init__.py
    metric.py
▾ networks/
    __init__.py
    lr_schedule.py
    network.py
▾ utils/
    __init__.py
    plot.py
__init__.py 
config.py
inference.py
README.md
train.py
```
## 网络结构
* ResNet+avgpool+(l2_normal+dropout+fc1)+(l2_normal+dropout+fc2)
* 损失函数: 交叉熵 Cross Entropy Loss
* 优化器: Adam

## RUN
* STEP0:
  ```
  git clone https://github.com/xungeer29/Remote-Sensing-Image-Classification
  cd Remote-Sensing-Image-Classification
  ```
* STEP1: 添加文件搜索路径

  将所有的`.py`文件的`sys.path.append`中添加的路径改为自己的项目路径
* STEP2: 划分训练集和本地验证集
  ```
  python dataset/create_img_list.py
  ```
* STEP3: train
  ```
  python train.py
  ```
* STEP4: test，并绘制混淆矩阵
  ```
  python inference.py
  ```
* STEP5: 使用log绘制精度曲线
  ```
  python utils/plot.py
  ```


## Results
* 全部重新训练，所有层相同的lr，acc@top1 = 0.65
* 冻结所有卷积层，只训练FC，acc@top1 = 0.926877
* 冻结ResNet的前三个layer，训练layer4与FC，acc@top1 = 97.8774
* 这种纯净的、数据分布完全平衡的数据集，仔细调一调是可以达到无限接近100%的准确率的
