# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/media/gfx/data1/DATA/lida/UCMerced_LandUse/UCMerced_LandUse/Images' # 数据集的根目录
    model = 'ResNet152' # ResNet34 使用的模型

    seed = 1000 # 固定随机种子
    num_workers = 12 # DataLoader 中的多线程数量
    num_classes = 21 # 分类类别数
    num_epochs = 300
    batch_size = 16
    lr = 0.001 # 初始lr
    width = 256 # 输入图像的宽
    height = 256 # 输入图像的高
    iter_smooth = 10 # 打印&记录log的频率

    checkpoint = 'ResNet152.pth' # 训练完成的模型名

config = DefaultConfigs()
