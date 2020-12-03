import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
class HSIDataset(Dataset):
    def __init__(self, data, label, n_components=1, patchsz=1):
        '''
        :param data: [h, w, bands]
        :param label: [h, w]
        :param n_components: scale
        :param patchsz: scale
        '''
        super(HSIDataset, self).__init__()
        self.data = data # [h, w, bands]
        self.label = label # [h, w]
        self.patchsz = patchsz
        # 原始数据的维度
        self.h, self.w, self.bands = self.data.shape
        self.Normalize()
        self.setPCA(self.data.reshape((self.h * self.w, self.bands)), n_components)
        # self.get_mean()
        # # 数据中心化
        # self.data -= self.mean
        self.addMirror()

    # 计算投影矩阵
    def setPCA(self, data, n_components):
        self.pca = PCA(n_components)
        self.pca.fit(data)

    # 数据归一化
    def Normalize(self):
        data = self.data.reshape((self.h * self.w, self.bands))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        self.data = data.reshape((self.h, self.w, self.bands))

    # 添加镜像
    # 处理patchsz为偶数时的情况，中心像素点位于右下patchsz//2 * patchsz//2 矩形的左上角
    def addMirror(self):
        dx = self.patchsz // 2
        if dx != 0:
            mirror = np.zeros((self.h + 2 * dx - 1, self.w + 2 * dx - 1, self.bands))
            mirror[dx:dx + self.h, dx:dx + self.w, :] = self.data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                if dx - i != 1:
                    mirror[:, -i - 1, :] = mirror[:, -(2 * (dx - 1) - i) - 1, :]
                    mirror[-i - 1, :, :] = mirror[-(2 * (dx - 1) - i) - 1, :, :]
            self.data = mirror
    def __len__(self):
        return self.h * self. w

    def __getitem__(self, index):
        '''
        :param index:
        :return: 元素光谱信息， 元素的空间信息， 标签
        '''
        l = index // self.w
        c = index % self.w
        dx = self.patchsz
        # 领域
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # neighbor_region = self.data[]
        # 中心像素的光谱
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        # 类别
        # target = self.label[l + self.patchsz // 2, c + self.patchsz // 2]
        target = self.label[l, c]
        # 降维
        reduction = self.pca.transform(neighbor_region.reshape((self.patchsz**2, self.bands)))
        neighbor_region_pca = reduction.reshape((self.patchsz, self.patchsz, -1))
        # neighbor_region_pca = neighbor_region
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_pca, dtype=torch.float32)), \
        torch.tensor(target, dtype=torch.long)

class HSIDatasetV1(HSIDataset):
    def __init__(self, data, label, n_components=1, patchsz=1):
        super().__init__(data, label, n_components, patchsz)
        self.sampleIndex = list(zip(*np.nonzero(self.label)))

    def __len__(self):
        return len(self.sampleIndex)

    def __getitem__(self, index):
        l, c = self.sampleIndex[index]
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        target = self.label[l, c] - 1
        reduction = self.pca.transform(neighbor_region.reshape(self.patchsz**2, self.bands))
        neighbor_region_pca = reduction.reshape((self.patchsz, self.patchsz, -1))
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_pca, dtype=torch.float32)), \
                torch.tensor(target, dtype=torch.long)

class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt',
        'patchsz': 32,
        'hzOfSe': 128,
        'hzOfSa': 256
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'patchsz': 32,
            'hzOfSe': 128,
            'hzOfSa': 256
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'patchsz': 64,
            'hzOfSe': 128,
            'hzOfSa': 256
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
    }}

# 验证数据集是否读取正确
# from scipy.io import loadmat
# # KSC
# m = loadmat('data/KSC/KSC.mat')
# data = m['KSC']
# data = data.astype(np.float32)
# m = loadmat('data/KSC/KSC_gt.mat')
# target = m['KSC_gt']
# target = target.astype(np.long)
# dataset = HSIDataset(data, target, patchsz=32)
# index = 42318
# (spectra, neighbor), label = dataset[index]
# print(torch.equal(spectra, neighbor[16, 16]))
# h, w = target.shape
# l = index // w
# c = index % w
# print(label)
# print(target[l ,c])