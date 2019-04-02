# Remote-Sensing-Image-Classification
## 数据集
```
链接: https://pan.baidu.com/s/1Avcih8rARD2LoBp5S4n2ww 
提取码: hp5f
```
## ENVS
python==2.7
pytorch==0.4.1

## 文件说明
* config.py: 超参数

## RUN
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
* STEP4: inference
  ```
  python inference.py
  ```
