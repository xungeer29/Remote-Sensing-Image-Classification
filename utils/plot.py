# -*- coding:utf-8 -*-

"""
绘制模型迭代曲线图
"""
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }

def smooth(scalar, weight=0.9):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def draw_curve(y1, y2=None):
    x1 = [i for i in range(len(y1))]
    # y1 = smooth(y1, weight=0.95)
    plt.plot(x1, y1, color='b', label='train')
    if y2 is not None:
        x2 = [i for i in range(len(y2))]
        # y2 = smooth(y2, weight=0.95)
        plt.plot(x2, y2, color='coral', label='validation')
    plt.legend(loc='lower right', prop=font1, frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if not os.path.exists('./figs'):
        os.makedirs('./figs')
    plt.savefig('./figs/acc.jpg')
    plt.savefig('./figs/acc.eps')

def plot_confusion_matrix(y_true, y_pred, labels,title='Normalized confusion matrix'):
    cmap = plt.cm.Blues
    ''' 颜色参考http://blog.csdn.net/haoji007/article/details/52063168'''
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 9), dpi=360)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=9, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=9, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=9, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('./figs/confusion_matrix.jpg', dpi=300)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    train = [0]
    val = [0]
    sum_train_acc = 0
    with open('./log/log.txt') as f:
        for i, line in enumerate(f.readlines()):
            if i < 8:
                continue
            elif i < 5311:
                if ((i-7)%11) == 0:
                    val_acc = line.strip().split(' ')[-4]
                    val.append(float(val_acc[:-1]))

                    avg_train_acc = sum_train_acc / 10
                    train.append(avg_train_acc)
                    sum_train_acc = 0
                else:
                    train_acc = float(line.strip().split(' ')[-1])
                    sum_train_acc += train_acc

    draw_curve(val[:81], train[:81])
