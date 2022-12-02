from torchvision import transforms
import torch
import torch.nn.functional as F 
from torch import nn 
from PIL import Image
import os
import cv2
import time
import glob
from basivsr import BasicVSR
from memory_profiler import profile



def input_lrs(lrs,model,device,is_use_cuda=False):
    '''
        Input random lrs, and return procced lrs
        Args:
            lrs: input random frames
            model: BasicVSR net
            device: server device
            is_use_cuda: use GPU or not (default: False)
    '''
    if is_use_cuda&(device==torch.device("cuda")):#使用cuda，且cuda可用
        lrs=lrs.to(device)
        model=model.to(device)
    rlt,time_consuming = model(lrs) 
    print(rlt.size())
    return time_consuming

def input_image(model,device,is_use_cuda=False):
    '''
        Input low quilty image, and return high quilty image
        Args:
            model: BasicVSR net
            device: server device
            is_use_cuda: use GPU or not (default: False)
    '''
    lrs=Image.open("/home/mawenzhuo/BasicVSR/images/ant.png")
    tran_totensor=transforms.ToTensor()
    lrs=tran_totensor(lrs)
    c,h,w=lrs.size()
    lrs=torch.reshape(lrs,[1,1,c,h,w])
    if is_use_cuda&(device==torch.device("cuda")):#使用cuda，且cuda可用
        lrs=lrs.to(device)
        model=model.to(device)
    rlt = model(lrs)
    print('HQ image size:',rlt.size())
    rlt=torch.reshape(rlt,[3,h*4,w*4])
    tran_PIL=transforms.ToPILImage()
    rlt=tran_PIL(rlt)
    rlt.save("/home/mawenzhuo/BasicVSR/images/ant_r.png")

def input_vedio(model,device,is_use_cuda=False):
    '''
        Input low quilty video, and return high quilty video
        Args:
            model: BasicVSR net
            device: server device
            is_use_cuda: use GPU or not (default: False)
    '''
    #将视频转为一帧帧图片
    cap = cv2.VideoCapture("/home/mawenzhuo/BasicVSR/frames/video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    frames=[]
    while suc:
        suc, frame = cap.read()
        if suc==False:
            break
        cv2.imwrite('/home/mawenzhuo/BasicVSR/frames/frame{}.png'.format(frame_count), frame)
        frames.append(frame)
        frame_count += 1
    cap.release()
    #格式转换
    lrs=torch.tensor(frames)
    lrs=lrs.unsqueeze(0)
    lrs=lrs.permute([0,1,4,2,3])
    if is_use_cuda&(device==torch.device("cuda")):#使用cuda，且cuda可用
        lrs=lrs.to(device)
        model=model.to(device)
    print('LQ video size: ',lrs.size())
    rlt=model(lrs)#进行超分
    print('HQ video size: ',rlt.size())
    #将视频帧拼接成视频
    rlt=rlt.permute([0,1,4,3,2]).detach().cpu().numpy().astype('uint8')
    f = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter('/home/mawenzhuo/BasicVSR/frames/video_r.mp4',f,fps,(rlt.shape[2],rlt.shape[3]))
    for i in range(frame_count):
        frame=rlt[0,i,:,:,:]
        videoWriter.write(frame)
    videoWriter.release()

def test1(model,device):
    '''
        performance test1
    '''
    time_cpu=[]
    time_cuda=[]
    time_cudnn=[]
    for num in range(1,62,3):
        lrs1 = torch.randn(1, num, 3, 64, 64)
        #cpu运算
        start1=time.time()
        input_lrs(lrs1,model,device,False)
        end1=time.time()
        time1=end1-start1
        #存储时间
        time_cpu.append(time1)
    for num in range(1,62,3):
        lrs2 = torch.randn(1, num, 3, 64, 64)
        #gpu加速
        start2=time.time()
        input_lrs(lrs2,model,device,True)
        end2=time.time()
        time2=end2-start2
        #存储时间
        time_cuda.append(time2)
    for num in range(1,62,3):
        lrs3 = torch.randn(1, num, 3, 64, 64)
        #cuDNN加速
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        start3=time.time()
        input_lrs(lrs3,model,device,True)
        end3=time.time()
        time3=end3-start3
        #存储时间
        time_cudnn.append(time3)
    file = open(r"BasicVSR/result/result2.txt", "w", encoding="utf-8")
    for i in range(len(time_cpu)):
        file.write('{} {} {} {}\n'.format(1+3*i,time_cpu[i],time_cuda[i],time_cudnn[i]))
    file.close()

def test2(model,device):
    '''
        performance test2
    '''
    time_cpu=[]
    time_cuda=[]
    time_cudnn=[]
    for num in range(10):
        lrs1 = torch.randn(1, 30, 3, 64, 64)
        #cpu运算
        start1=time.time()
        input_lrs(lrs1,model,device,False)
        end1=time.time()
        time1=end1-start1
        #存储时间
        time_cpu.append(time1)
    for num in range(10):
        lrs2 = torch.randn(1, 30, 3, 64, 64)
        #gpu加速
        start2=time.time()
        input_lrs(lrs2,model,device,True)
        end2=time.time()
        time2=end2-start2
        #存储时间
        time_cuda.append(time2)
    for num in range(10):
        lrs3 = torch.randn(1, 30, 3, 64, 64)
        #cuDNN加速
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        start3=time.time()
        input_lrs(lrs3,model,device,True)
        end3=time.time()
        time3=end3-start3
        #存储时间
        time_cudnn.append(time3)
    file = open(r"BasicVSR/result/result2.txt", "w", encoding="utf-8")
    for i in range(len(time_cpu)):
        file.write('{} {} {} {}\n'.format(i,time_cpu[i],time_cuda[i],time_cudnn[i]))
    file.close()


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicVSR(spynet_path="/home/mawenzhuo/BasicVSR/spynet_weight.pth")#spynet_path="/home/mawenzhuo/BasicVSR/spynet_weight.pth"
    lrs = torch.randn(1, 40, 3, 64, 64)
    start=time.time()
    input_lrs(lrs,model,device,True)
    # input_image(model,device,True)
    # input_vedio(model,device,True)
    end=time.time()
    print('The total time consumption is {} s'.format(end-start))
    #test1(model,device)
    #test2(model,device)
    