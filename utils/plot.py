# -*- coding:utf-8 -*-

"""
绘制模型迭代曲线图
"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 9,
         }

def smooth(scalar, weight=0.9):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def draw_curve(y1, name, y2=None):
    x1 = [i for i in range(len(data1))]
    y1 = smooth(y1, weight=0.95)
    plt.plot(x, y1, color='b', label='train')
    if y2 not is None:
        y2 = smooth(y2, weight=0.95)
