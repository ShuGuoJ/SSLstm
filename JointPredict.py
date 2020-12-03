import torch
import numpy as np
from scipy.io import loadmat, savemat
from utils import loadLabel
from HSIDataset import HSIDatasetV1, DatasetInfo
import os
from torch.utils.data import DataLoader
from Model.module import SeLstm, SaLstm
import math
isExist = lambda path: os.path.exists(path)
DETA_EPOCH = 5
SAMPLE_PER_CLASS = [10, 50, 100]
RUN = 10
DATASETNAME = ['Salinas', 'PaviaU', 'KSC']
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# 获得准确率最大的临近批次
def getEpochOfMaxAcc(acc):
    epoch = int(np.argmax(acc))
    epoch =  epoch + DETA_EPOCH - epoch % DETA_EPOCH
    return min(epoch, 995)


# 联合预测
def jointPredict(seLstm, saLstm, loader):
    seLstm.eval()
    saLstm.eval()
    seLstm.to(DEVICE)
    saLstm.to(DEVICE)
    correct = 0
    for (spectra, neighbor_region), target in loader:
        spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        spectra_logits, spatia_logits = seLstm(spectra), saLstm(neighbor_region)
        joint_logits = 0.5 * spectra_logits + 0.5 * spatia_logits
        pred = torch.argmax(joint_logits, dim=-1)
        correct += torch.sum(pred == target).item()
    return correct / len(loader.dataset)


def main(datasetName, sample_per_class, run):
    # 加载数据和标签
    info = DatasetInfo.info[datasetName]
    data_path = "data/{0}/{0}.mat".format(datasetName)
    assert isExist(data_path)
    data = loadmat(data_path)[info['data_key']]
    label_path = "trainTestSplit/{}/sample{}_run{}.mat".format(datasetName, sample_per_class, run)
    assert isExist(label_path)
    _, testLabel = loadLabel(label_path)
    # 数据转换
    data, testLabel = data.astype(np.float32), testLabel.astype(np.int)
    # 转换为数据集
    dataset = HSIDatasetV1(data, testLabel, patchsz=info['patchsz'])
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
    bands = data.shape[-1]
    nc = np.max(testLabel)
    # 定义模型
    seLstm = SeLstm(1, info['hzOfSe'], nc)
    saLstm = SaLstm(info['patchsz'], info['hzOfSa'], nc)
    # 加载预训练模型
    seRoot = "SeLstm/{}/{}/{}".format(datasetName, sample_per_class, run)
    acc = loadmat(os.path.join(seRoot, "res.mat"))['acc']
    seEpoch = getEpochOfMaxAcc(acc)
    saRoot = "SaLstm/{}/{}/{}".format(datasetName, sample_per_class, run)
    acc = loadmat(os.path.join(saRoot, 'res.mat'))['acc']
    saEpoch = getEpochOfMaxAcc(acc)
    seLstm.load_state_dict(torch.load(os.path.join(seRoot, "SeLstm_sample{}_run{}_epoch{}.pkl".format(sample_per_class, run,
                                                                                                      seEpoch))))
    saLstm.load_state_dict(torch.load(os.path.join(saRoot, "SaLstm_sample{}_run{}_epoch{}.pkl".format(sample_per_class, run,
                                                                                                      saEpoch))))
    return jointPredict(seLstm, saLstm, loader)


if __name__ == "__main__":
    for datasetName in DATASETNAME:
        print("*"*5 + datasetName + "*"*5)
        for sample_per_class in SAMPLE_PER_CLASS:
            res = np.zeros((RUN))
            print("*"*5 + str(sample_per_class) + "*"*5)
            for r in range(RUN):
                acc = main(datasetName, sample_per_class, r)
                res[r] = acc
                print("run:{} acc:{:.2%}".format(r, acc))
            root = "joint/{}/{}".format(datasetName, sample_per_class)
            if not os.path.isdir(root):
                os.makedirs(root)
            savemat(os.path.join(root, "res.mat"), {"acc":res})


