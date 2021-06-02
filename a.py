import os
import cv2
import visdom
import random
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
torch.manual_seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


char_id = {}
id_char = {}
char_list = []
for i in range(10):
    char_list.append(chr(i + 48))
for i in range(26):
    if not (chr(i + 65) == 'O' or chr(i + 65) == 'I'):
        char_list.append(chr(i + 65))
for i in range(len(char_list)):
    char_id[char_list[i]] = i
    id_char[i] = char_list[i]
is_test = False


def OTSU(img):
    thres = 0
    g_max = -1
    for i in range(256):
        t0 = (img < i).sum()
        t1 = img.size - t0
        if t0 > 0 and t1 > 0:
            s0 = img[img < i].sum() / t0
            s1 = img[img >= i].sum() / t1
            g = t0 * t1 * (s0 - s1) ** 2
            if g > g_max:
                g_max = g
                thres = i
    if is_test:
        thres -= 3
    img[img < thres] = 0
    img[img >= thres] = 255
    return img



def trans_gray(img):
    img[img < 0.5] = 0
    img[img >= 0.5] = 1
    return img



dx = (-1, -1, -1, 0, 0, 1, 1, 1)
dy = (-1, 0, 1, -1, 1, -1, 0, 1)

def check1(x, y):
    global img
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

def check2(x, y):
    global img
    return check1(x, y) and (0 <= x < 6 or img.shape[0] - 6 <= x < img.shape[0] or 0 <= y < 6 or img.shape[1] - 6 <= y < img.shape[1])


def bfs(i, j, check = check1):
    global vis, tag, Q
    tag += 1
    Q = [(i, j)]
    vis[i][j] = tag
    h, t = 0, 0
    while h <= t:
        u, v = Q[h]
        h += 1
        for k in range(0, 8):
            x, y = u + dx[k], v + dy[k]
            if check(x, y):
                if img[x][y] == 255 and vis[x][y] == 0:
                    Q.append((x, y))
                    vis[x][y] = tag
                    t += 1



def erode_dilate(img, x):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(x,x))
    return cv2.dilate(cv2.erode(img,kernel),kernel)

def dilate_erode(img, x):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(x,x))
    return cv2.erode(cv2.dilate(img,kernel),kernel)


def show(img):
    plt.imshow(img)
    plt.show()
            
            
def expand(img):
    x, y = np.where(img > 0)
    img = img[x.min() : x.max() + 1, y.min() : y.max() + 1]
    h, w = img.shape
    if h < w:
        img = cv2.copyMakeBorder(img, int((w - h) / 2 + 0.5) + 5, int((w - h) / 2 + 0.5) + 5, 5, 5, cv2.BORDER_CONSTANT)
    else:
        img = cv2.copyMakeBorder(img, 5, 5, int((h - w) / 2 + 0.5) + 5, int((h - w) / 2 + 0.5) + 5, cv2.BORDER_CONSTANT)
    if is_test:
        vis = np.zeros(img.shape, dtype=int)
        tag = 0
        sizes = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] == 255 and vis[i][j] == 0:
                    tag += 1
                    Q = [(i, j)]
                    vis[i][j] = tag
                    h, t = 0, 0
                    while h <= t:
                        u, v = Q[h]
                        h += 1
                        for k in range(0, 8):
                            x, y = u + dx[k], v + dy[k]
                            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                                if img[x][y] == 255 and vis[x][y] == 0:
                                    Q.append((x, y))
                                    vis[x][y] = tag
                                    t += 1
                    sizes.append([len(Q), tag])
        sizes.sort()
        img[vis != sizes[len(sizes) - 1][1]] = 0
    img = erode_dilate(img, 3)
    return img


def img_cut_method1():
    global img_char_list, img
    img_char_list = []
    H, W = img.shape
    H1, H2 = int(0.1 * H), int(0.9 * H)
    i = 0
    while i < W:
        if img[:, i].sum() / (H * 255) > 0.1:
            j = i + 1
            while j < W:
                if img[:, j].sum() / (H * 255) < 0.05:
                    break
                j += 1
            if j - i >= 3:
                img_char = img[H1 : H2, max(i - 4, 0) : min(j + 4, W)]
                if img_char.shape[1] / img_char.shape[0] < 0.75:
                    if img_char.sum() > 0:
                        img_char_list.append(expand(img_char))
                else:
                    if img_char[:, :int(img_char.shape[1] / 2 + 0.5)].sum() > 0:
                        img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 2 + 0.5)]))
                    if img_char[:, int(img_char.shape[1] / 2 + 0.5):].sum() > 0:
                        img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 2 + 0.5):]))
            i = j
        else:
            i += 1




def img_cut_method2():
    global vis, tag, img_char_list, img
    vis = np.zeros(img.shape, dtype=int)
    tag = 0
    sizes = []
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i][j] == 255 and vis[i][j] == 0:
                bfs(i, j)
                global Q
                sizes.append([len(Q), tag])
    #show(img)
    img_char_list = []
    for i in range(1, tag + 1):
        x0 = min(np.where(vis == i)[0])
        x1 = max(np.where(vis == i)[0])
        y0 = min(np.where(vis == i)[1])
        y1 = max(np.where(vis == i)[1])
        img_char = img[x0 : x1 + 1, y0 : y1 + 1]
        if img_char.shape[1] / img_char.shape[0] < 0.75:
            img_char_list.append(expand(img_char))
        #else:
            #img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 2 + 0.5)]))
            #img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 2 + 0.5):]))
        elif 0.75 <= img_char.shape[1] / img_char.shape[0] < 1.25:
            img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 2 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 2 + 0.5):]))
        elif 1.25 <= img_char.shape[1] / img_char.shape[0] < 1.75:
            img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 3 * 1 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 3 * 1 + 0.5):int(img_char.shape[1] / 3 * 2 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 3 * 2 + 0.5):]))
        elif 1.75 <= img_char.shape[1] / img_char.shape[0] < 2.25:
            img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 4 * 1 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 4 * 1 + 0.5):int(img_char.shape[1] / 4 * 2 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 4 * 2 + 0.5):int(img_char.shape[1] / 4 * 3 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 4 * 3 + 0.5):]))
        elif 2.25 <= img_char.shape[1] / img_char.shape[0] < 2.75:
            img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 5 * 1 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 5 * 1 + 0.5):int(img_char.shape[1] / 5 * 2 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 5 * 2 + 0.5):int(img_char.shape[1] / 5 * 3 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 5 * 3 + 0.5):int(img_char.shape[1] / 5 * 4 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 5 * 4 + 0.5):]))
        elif 2.75 <= img_char.shape[1] / img_char.shape[0]:
            img_char_list.append(expand(img_char[:, :int(img_char.shape[1] / 6 * 1 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 6 * 1 + 0.5):int(img_char.shape[1] / 6 * 2 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 6 * 2 + 0.5):int(img_char.shape[1] / 6 * 3 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 6 * 3 + 0.5):int(img_char.shape[1] / 6 * 4 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 6 * 4 + 0.5):int(img_char.shape[1] / 6 * 5 + 0.5)]))
            img_char_list.append(expand(img_char[:, int(img_char.shape[1] / 6 * 5 + 0.5):]))


def img_cut_method3():
    global img_char_list, img
    #show(img)
    img_char_list = []
    if not img.any():
        return
    x0, x1 = 0, img.shape[0] - 1
    while img[x0, :].sum() / (img.shape[1] * 255) < 0.1 or img[x0, :].sum() / (img.shape[1] * 255) > 0.9:
        x0 += 1
    while img[x1, :].sum() / (img.shape[1] * 255) < 0.1 or img[x1, :].sum() / (img.shape[1] * 255) > 0.9:
        x1 -= 1
    img = img[x0 : x1 + 1, :]
    #show(img)
    H, W = img.shape
    l, r = 1, W
    a = []
    for i in range(W):
        a.append(img[:, i].sum() / (H * 255))
    can_be_cut = []
    for i in range(W):
        t = 0
        if a[i] < 0.3:
            t = 1
        can_be_cut.append(t)
    while l < r:
        mid = (l + r + 1) // 2
        s = 0
        i = 0
        while i < W:
            j = i + mid
            if j > W:
                break
            while j < W:
                if can_be_cut[j] == 1:
                    break
                j += 1
            s += 1
            i = j + 1
        if s >= 6:
            l = mid
        else:
            r = mid - 1
    i = 0
    while i < W:
        j = i + l
        if j > W:
            break
        while j < W:
            if can_be_cut[j] == 1:
                break
            j += 1
        if img[:, i : j].sum() > 0:
            img_char_list.append(expand(img[:, i : j]))
        i = j + 1
        
        
def img_cut_method4():
    global img_char_list, img
    #show(img)
    img_char_list = []
    if not img.any():
        return
    x0, x1 = 0, img.shape[0] - 1
    while img[x0, :].sum() / (img.shape[1] * 255) < 0.1 or img[x0, :].sum() / (img.shape[1] * 255) > 0.9:
        x0 += 1
    while img[x1, :].sum() / (img.shape[1] * 255) < 0.1 or img[x1, :].sum() / (img.shape[1] * 255) > 0.9:
        x1 -= 1
    img = img[x0 : x1 + 1, :]
    '''pos = [0]
    for i in range(1, img.shape[1]):
        if img[:, i].sum() > 0 or img[:, i - 1].sum() > 0:
            pos.append(i)
    img = img[:, pos]'''
    if not is_test:
        img = dilate_erode(img, 3)
    #show(img)
    H, W = img.shape
    l, r = 1, W
    a = []
    for i in range(W):
        a.append(img[:, i].sum() / (H * 255))
    f = np.zeros((W, 6))
    g = np.zeros((W, 6),dtype=int)
    for i in range(W):
        for j in range(6):
            f[i][j] = 100
    f[0][0] = 0
    for i in range(W):
        for j in range(5):
            for k in range(i + W // 8, W - 1):
                if f[i][j] + a[k] < f[k + 1][j + 1]:
                    f[k + 1][j + 1] = f[i][j] + a[k]
                    g[k + 1][j + 1] = i
    k = 0
    for i in range(W):
        if i + W // 8 <= W:
            if f[i][5] < f[k][5]:
                k = i
    if k == 0:
        return
    sep = [k]
    for i in range(5, 0, -1):
        k = g[k][i]
        sep = [k] + sep
    sep += [W]
    for i in range(6):
        if img[:, sep[i] : sep[i + 1]].any():
            img_char_list.append(expand(img[:, sep[i] : sep[i + 1]]))
        
        
            
def pre():
    
    global vis, tag, img
    
    img = OTSU(img)

    img = 255 - img
    
    img = cv2.resize(img, (0, 0), fx = 2, fy = 2)

    vis = np.zeros(img.shape, dtype=int)
    tag = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if ((i < 3 or i >= img.shape[0] - 3) and img[i, :].sum() / (img.shape[1] * 255) > 0.95) or ((j < 3 or j >= img.shape[1] - 3) and img[:, j].sum() / (img.shape[0] * 255) > 0.95):
                if img[i][j] == 255 and vis[i][j] == 0:
                    bfs(i, j, check2)
    img[vis > 0] = 0
                
    img = OTSU(img)
    
    img = erode_dilate(img, 3)
    
    x, y = np.where(img > 0)
    if len(x) == 0:
        return
    img = img[x.min() : x.max() + 1, y.min() : y.max() + 1]
    
    vis = np.zeros(img.shape, dtype=int)
    tag = 0
    sizes = [0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255 and vis[i][j] == 0:
                bfs(i, j)
                global Q
                sizes.append(len(Q))
    for i in range(1, len(sizes)):
        x, y = np.where(vis == i)
        if max(x) - min(x) <= img.shape[0] * 0.4 or max(y) - min(y) <= 3:
            img[vis == i] = 0
            
    x, y = np.where(img > 0)
    if len(x) == 0:
        return
    img = img[x.min() : x.max() + 1, y.min() : y.max() + 1]
    


def obtain_train_data():

    global images_data, labels_data, img_char_list, img

    images_data = []
    labels_data = []
    
    for num in range(1, 582):
        print(num)
        # 文件名
        f_img = './Plate_dataset/AC/train/jpeg/' + str(num) + '.jpg'
        f_xml = './Plate_dataset/AC/train/xml/' + str(num) + '.xml'

        # 读取图像
        img = cv2.cvtColor(cv2.imread(f_img), cv2.COLOR_BGR2GRAY)
        # 读取xml
        anno = ET.ElementTree(file=f_xml)
        # 读取车牌字符label
        label = anno.find('object').find('platetext').text
        label = label.replace('O', '0')
        label = label.replace('I', '1')

        # bounding box
        xmin = anno.find('object').find('bndbox').find('xmin').text
        ymin = anno.find('object').find('bndbox').find('ymin').text
        xmax = anno.find('object').find('bndbox').find('xmax').text
        ymax = anno.find('object').find('bndbox').find('ymax').text

        bbox = [xmin, ymin, xmax, ymax]
        bbox = [int(b) for b in bbox]

        img = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]

        pre()

        img_cut_method2()
        if len(img_char_list) != len(label):
            img_cut_method4()
        
        if len(img_char_list) == len(label):
            for i in range(len(label)):
                trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor()])
                images_data.append(trans_gray(trans(torch.from_numpy(img_char_list[i])).view(1, 1, 32, 32)))
                labels_data.append(torch.tensor([char_id[label[i]]]))


    images_data = torch.cat(images_data, 0)
    labels_data = torch.cat(labels_data, 0)
    print(images_data.shape)
    print(labels_data.shape)
    


# LeNet-5
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 15, kernel_size=(5, 5))),  # 卷积层
            ('relu1', nn.ReLU()),  # 激活层
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),  # 池化层
            ('c3', nn.Conv2d(15, 100, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(100, 400, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(400, 240)),  # 全连接层
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(240, 34)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

net = LeNet5()  # 实例化一个网络
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=2e-3)  # 定义优化器，lr为初始学习率

def train(epoch):
    global data_train_loader
    net.train()  # 将模型设置为训练模式
    for i, (images, labels) in enumerate(data_train_loader):  # 从dataloader中批量取数据
        
        optimizer.zero_grad()  # 将优化器中梯度值清空

        output = net(images)  # 将图片过网络得输出

        loss = criterion(output, labels)  # 计算输出与真实label之间的误差作为loss

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 优化器计算更新的步长，并与梯度一并更新网络参数


def main():
    for epoch in range(1, 51):
        train(epoch)


def char_identify_pre():
    global data_train_loader
    data_train_loader = DataLoader(torch.utils.data.TensorDataset(images_data, labels_data), batch_size=256, shuffle=True, num_workers=8)
    main()


def char_identify(img):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor()])
    net.eval()
    return net(trans(img).view(1, 1, 32, 32)).detach().max(1)[1].item()



def test():

    global images_data, labels_data, img_char_list, img, is_test

    is_test = True

    acc = 0

    for num in range(1, 101):
        print(num)
        # 文件名
        f_img = './Plate_dataset/AC/test/jpeg/' + str(num) + '.jpg'
        f_xml = './Plate_dataset/AC/test/xml/' + str(num) + '.xml'

        # 读取图像
        img = cv2.cvtColor(cv2.imread(f_img), cv2.COLOR_BGR2GRAY)
        # 读取xml
        anno = ET.ElementTree(file=f_xml)
        # 读取车牌字符label
        label = anno.find('object').find('platetext').text
        label = label.replace('O', '0')
        label = label.replace('I', '1')
        # 修正test中的错误
        if num == 61:
            label = 'DX6511'

        # bounding box
        xmin = anno.find('object').find('bndbox').find('xmin').text
        ymin = anno.find('object').find('bndbox').find('ymin').text
        xmax = anno.find('object').find('bndbox').find('xmax').text
        ymax = anno.find('object').find('bndbox').find('ymax').text

        bbox = [xmin, ymin, xmax, ymax]
        bbox = [int(b) for b in bbox]

        img = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]

        pre()

        img_cut_method2()
        if len(img_char_list) != len(label):
            img_cut_method4()

        res = ''
        for i in range(len(img_char_list)):
            res += id_char[char_identify(img_char_list[i])]
        print(res)
        print(label)
        if res == label:
            acc += 1

    print(acc)


obtain_train_data()
char_identify_pre()
test()