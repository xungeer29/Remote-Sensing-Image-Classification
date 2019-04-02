class DefaultConfigs(object):
    data_root = '/media/gfx/data1/DATA/lida/UCMerced_LandUse/UCMerced_LandUse/Images'
    model = 'ResNet152' # ResNet34

    seed = 1000
    num_workers = 12
    num_classes = 21
    num_epochs = 10
    lr = 0.001
    width = 256
    height = 256



config = DefaultConfigs()
