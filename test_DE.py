import os

import numpy as np
import geatpy as ea
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt

import my_function
from data_set.figsettings import plt_head


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, data, initpars):
        name = 'ZDT1'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(initpars)  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = []  # 决策变量下界
        ub = []  # 决策变量上界
        for i in range(Dim):
            lb.append(initpars[i][0])
            ub.append(initpars[i][1])
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        self.initpars = initpars
        self.phase = data[:, 0]
        self.datay = data[:, 1] - np.mean(data[:, 1])
        self.cost = []
        #self.my_plt()   #展示数据量减少效果

        self.xspace = np.linspace(0, 1, 100)
        self.sigma = np.diff(self.datay, 2).std() / np.sqrt(6)  # Estimated observation noise values
        if Dim == 7:
            self.model = load_model('model10mc.hdf5')
        elif Dim == 8:
            self.model = load_model('model10l3mc.hdf5')
        else:
            raise RuntimeError('Input init-parameters should be with length of 6 or 7.')
        # results
        self.DE_trace = []

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def predict(self, allpara):
        arraymcn = np.array(allpara)
        if len(arraymcn) == 7:
            arraymc = arraymcn[0:5]
            mcinput = np.reshape(arraymc, (1, 5))
            lightdata = self.model(mcinput)
            proto_tensor = tf.make_tensor_proto(lightdata)
            data_numpy = tf.make_ndarray(proto_tensor)
            return data_numpy[0] + arraymcn[6]
        else:
            arraymc = arraymcn[0:6]
            mcinput = np.reshape(arraymc, (1, 6))
            lightdata = self.model(mcinput)
            proto_tensor = tf.make_tensor_proto(lightdata)
            data_numpy = tf.make_ndarray(proto_tensor)
            return data_numpy[0] + arraymcn[7]

    def getdata(self, allpara):
        arraymc = np.array(allpara)

        if len(arraymc) == 7:
            _offset = int(arraymc[5])
        else:
            _offset = int(arraymc[6])
        datay_m = np.hstack((self.datay[_offset:], self.datay[:_offset]))
        noise_y = np.interp(self.xspace, self.phase, datay_m)  # y轴
        return noise_y

    def get_lcfit(self, phen):
        return self.predict(phen)

    def aimFunc(self, pop):  # 目标函数
        cost = []
        x = pop.Chrom
        self.DE_trace.append(x)
        for i in range(pop.sizes):
            output = self.predict(x[i])
            noisey = self.getdata(x[i])
            # Calculating the likelihood function
            cost.append(0.5 * np.sum(np.log(2 * np.pi * self.sigma ** 2) + (output - noisey) ** 2 / (self.sigma ** 2)))
        pop.ObjV = np.reshape(np.array(cost), (-1, 1))  # 计算目标函数值，赋值给pop种群对象的ObjV属性

def test_geat(data, initpars, MAXGEN, NIND, Encoding):
    problem = MyProblem(data, initpars)  # 生成问题对象
    # 构建算法
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    myAlgorithm = ea.soea_DE_best_1_bin_templet(problem, population)  # 实例化一个算法模板对象
    # 求解
    myAlgorithm.MAXGEN = MAXGEN  # 最大进化代数
    myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 3  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    phen = BestIndi.Phen[0]

    pre = problem.get_lcfit(phen)

    if phen.shape[0] == 7:
        offset = int(phen[5])
    else:
        offset = int(phen[6])
    plt.figure()
    ax = plt.gca()
    datay = np.hstack((problem.datay[offset:], problem.datay[:offset]))


    ax.plot(problem.phase, problem.datay, '.', c='k')

    ax.plot(problem.xspace, pre, '-r')
    ax.yaxis.set_ticks_position('left')
    ax.invert_yaxis()  # y-axis reversed
    plt.xlabel('phase', fontsize=18)
    plt.ylabel('mag', fontsize=18)
    my_function.plt_minor_locator(ax)
    plt.show()
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    return BestIndi,population


def parameters(initpars, MAXGEN, NIND):
    path = '../MCMCNN-main/'
    file = 'KIC 6431545.txt'
    data = np.loadtxt(path + file)
    BestIndi,population = test_geat(data, initpars, MAXGEN, NIND, Encoding)


if __name__ == '__main__':
    plt_head()
    MAXGEN = 100  # 最大进化代数。
    NIND = 30  # 种群规模
    Encoding = 'RI'  # 编码方式

    initpars = [(4655 / 5850 - 0.8, 4655 / 5850 + 0.8),
                (54.40 / 90 - 90 / 90, 54.40 / 90 + 90 / 90),
                (0, 5),
                (0, 1.9),
                (0.1, 1.5),
                (0, 5),  # l3 par
                (-10, 10),
                (-0.1, 0.1)]
    parameters(initpars, MAXGEN, NIND)