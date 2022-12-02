import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 500 #图片像素
plt.rcParams['figure.dpi'] = 500 #分辨率
# Arial Helvetica TIMES NEW ROMAN #如果要显示中文字体，则在此处设为：SimHei
plt.rcParams['axes.unicode_minus']=False  #显示负号
import pandas as pd
import matplotlib.colors as mcolors
# color pallette 颜色盘
b1,color1,color2,color3,color4,color5 = '#636efa','#c8141c','dodgerblue', '#8714d0', 'green', 'orangered' 


def analyse1():
    #数据读取
    time=[[],[],[],[],[],[]]
    file=open(r'BasicVSR/result/result1.txt','r',encoding='utf-8')
    while True:
        line = file.readline()
        if not line: # 为None跳出
            break
        line=line.splitlines()
        line=line[0].split(' ')
        time[0].append(int(line[0]))
        time[1].append(float(line[1]))
        time[2].append(float(line[2]))
        time[3].append(float(line[3]))
        time[4].append(float(line[1])/float(line[2]))
        time[5].append(float(line[1])/float(line[3]))
    file.close()
    plt.figure(figsize=(10,5))
    #plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    # ax.tick_params(axis=u'both', which=u'both',length=0) # 去掉刻度线
    plt.plot(time[0],time[1],label='CPU',linewidth=2,color=color1)
    plt.plot(time[0],time[2],label='CUDA',linewidth=2,color=color2)
    plt.plot(time[0],time[3],label='cuDNN',linewidth=2,color=color3)
    plt.legend()#显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.xticks(fontsize=12,fontweight='bold') #默认字体大小为10
    plt.yticks(fontsize=12,fontweight='bold')
    plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细
    plt.xlabel("Frame Number",fontsize=13,fontweight='bold')
    plt.ylabel("Time Consuming",fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.xlim(0,65)#设置x轴的范围
    plt.ylim(0,15)
    plt.savefig('BasicVSR/result/analyse1.png',format='png')

    plt.figure(figsize=(10,5))
    #plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    # ax.tick_params(axis=u'both', which=u'both',length=0) # 去掉刻度线
    plt.plot(time[0],time[4],label='CUDA',linewidth=2,color=color1)
    plt.plot(time[0],time[5],label='cuDNN',linewidth=2,color=color2)
    plt.legend()#显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.xticks(fontsize=12,fontweight='bold') #默认字体大小为10
    plt.yticks(fontsize=12,fontweight='bold')
    plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细
    plt.xlabel("Frame Number",fontsize=13,fontweight='bold')
    plt.ylabel("Speed-up Ratio",fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.xlim(0,65)#设置x轴的范围
    plt.ylim(0,10)
    plt.savefig('BasicVSR/result/analyse2.png',format='png')

def analyse2():
    time=[[],[],[],[]]
    file=open(r'BasicVSR/result/result2.txt','r',encoding='utf-8')
    while True:
        line = file.readline()
        if not line: # 为None跳出
            break
        line=line.splitlines()
        line=line[0].split(' ')
        time[0].append(int(line[0]))
        time[1].append(float(line[1]))
        time[2].append(float(line[2]))
        time[3].append(float(line[3]))
    file.close()
    plt.figure(figsize=(10,5))
    #plt.grid(linestyle = "--")      #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    # ax.tick_params(axis=u'both', which=u'both',length=0) # 去掉刻度线
    plt.plot(time[0],time[1],label='CPU',linewidth=2,color=color1)
    plt.plot(time[0],time[2],label='CUDA',linewidth=2,color=color2)
    plt.plot(time[0],time[3],label='cuDNN',linewidth=2,color=color3)
    plt.legend()#显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.xticks(fontsize=12,fontweight='bold') #默认字体大小为10
    plt.yticks(fontsize=12,fontweight='bold')
    plt.setp(ltext, fontsize=12,fontweight='bold') #设置图例字体的大小和粗细
    plt.xlabel("Number of Calculations",fontsize=13,fontweight='bold')
    plt.ylabel("Time Consuming",fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.xlim(0,10)#设置x轴的范围
    plt.ylim(0,10)
    plt.savefig('BasicVSR/result/analyse3.png',format='png')

def analyse3():
    #数据读取
    time=[[],[],[],[]]
    file=open('D:\\本科\\高性能地理计算\\基于CUDA的BasicVSR\\BasicVSR\\result\\result4.txt','r',encoding='utf-8')
    while True:
        line = file.readline()
        if not line: # 为None跳出
            break
        line=line.splitlines()
        line=line[0].split(' ')
        time[0].append(float(line[0]))
        time[1].append(float(line[1]))
        time[2].append(float(line[2]))
        time[3].append(float(line[3]))
    file.close()
    time=np.array(time)
    # 设置画布大小
    plt.figure(figsize=(12,6))
    figure,axes = plt.subplots(1,1,figsize = (6,6),dpi = 120)
    # 构造数据
    y = [np.average(time[0][0:3]),np.average(time[1][0:3]),np.average(time[2][0:3]),np.average(time[3][0:3])]
    label=['Alignment','Propagation','Aggregation','Upsampling']
    # 绘图
    plt.pie(y,labels=label,autopct='%.2f%%')
    # 添加大标题，并设置字号大小，以及定义所用字体
    plt.title("CPU",fontsize = 28)
    # 输出为常规的png格式
    plt.savefig(r"D:\\本科\\高性能地理计算\\基于CUDA的BasicVSR\\BasicVSR\\result\\analyse5.png", format="png")

analyse3()