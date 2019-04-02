# Remote-Sensing-Image-Classification
## ENVS
python==2.7
pytorch==0.4.1

## RUN
* STEP1: 

  将所有的`.py`文件的`sys.path.append`中添加的路径改为自己的项目路径
* 划分训练集和本地验证集
  ```
  python dataset/create_img_list.py
  ```
