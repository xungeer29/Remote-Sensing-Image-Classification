# Remote-Sensing-Image-Classification
## Dataset
一个纯净的、没有噪声的遥感图像数据集，共21类，每类100张图像，可以用于分类任务的入门练手

在本次的项目中，将数据集按照 8:2 随机划分为训练集和验证集

项目中包含了精度曲线绘制、log记录等，算是一套完整的pipleline，可以针对不同的任务进行快速更改

```
链接: https://pan.baidu.com/s/1Avcih8rARD2LoBp5S4n2ww 
提取码: hp5f
```
## ENVS
* python==2.7
* pytorch==0.4.1

## File Structure
```
remote_sensing_image_classification/ # 根目录
▾ data/
    label_list.txt # label
    train.txt # 训练集路径及标注
    valid.txt # 验证集路径及标注
▾ dataset/
    __init__.py
    create_img_list.py # 随机8:2划分数据集，生成 ./data/ 文件夹下的txt文件
    dataset.py # 数据读取
▾ figs/
    acc.eps # 精度曲线
    acc.jpg # 精度曲线 矢量图
    confusion_matrix.jpg # 混淆矩阵
▾ log/
    log.txt # 记录log
▾ metrics/
    __init__.py
    metric.py # 指标，主要是精度
▾ networks/
    __init__.py
    lr_schedule.py # 学习率的调整策略
    network.py # 网络结构
▾ utils/
    __init__.py
    plot.py # 绘制曲线
__init__.py 
config.py # 超参数的集合
inference.py # 推理，前向，用于测试
README.md # 说明
train.py # 训练&验证脚本
```
## Network Architecture
* ResNet+avgpool+(l2_normal+dropout+fc1)+(l2_normal+dropout+fc2)
* 损失函数: 交叉熵 Cross Entropy Loss
* 优化器: Adam

## RUN
* STEP0:
  ```
  git clone https://github.com/xungeer29/Remote-Sensing-Image-Classification
  cd Remote-Sensing-Image-Classification
  ```
* STEP1: 添加文件搜索路径，更改数据集根目录

  将所有的`.py`文件的`sys.path.append`中添加的路径改为自己的项目路径

  更改`config.py`中的`data_root`为数据集存放的根目录
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
