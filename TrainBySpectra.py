import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat, savemat
from utils import loadLabel, weight_init
from HSIDataset import HSIDatasetV1, DatasetInfo
from Model.module import SeLstm
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom
from train import train, test


isExists = lambda path: os.path.exists(path)
SAMPLE_PER_CLASS = [10, 50, 100]
RUN = 10
EPOCHS = 10
LR = 1e-1
BATCHSZ = 10
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom()
ROOT = None

def main(datasetName, n_sample_per_class, run, encoderPath=None):
    # 加载数据和标签
    info = DatasetInfo.info[datasetName]
    data_path = "./data/{}/{}.mat".format(datasetName, datasetName)
    label_path = './trainTestSplit/{}/sample{}_run{}.mat'.format(datasetName, n_sample_per_class, run)
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]
    bands = data.shape[2]
    isExists(label_path)
    trainLabel, testLabel = loadLabel(label_path)
    res = torch.zeros((3, EPOCHS))
    # 数据转换
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(np.int), testLabel.astype(np.int)
    nc = int(np.max(trainLabel))
    trainDataset = HSIDatasetV1(data, trainLabel, patchsz=info['patchsz'])
    testDataset = HSIDatasetV1(data, testLabel, patchsz=info['patchsz'])
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)
    model = SeLstm(1, info['hzOfSe'], nc)
    # 模型参数初始化
    model.apply(weight_init)
    # 加载编码器的预训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader, DEVICE=DEVICE)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader, DEVICE=DEVICE)
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        res[0][epoch], res[1][epoch], res[2][epoch] = trainLoss, evalLoss, acc
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(ROOT, 'SeLstm_sample{}_run{}_epoch{}.pkl'.format(n_sample_per_class,
                                                                                                       run, epoch)))
        # scheduler.step()
    tmp = res.numpy()
    savemat(os.path.join(ROOT, 'res.mat'), {'trainLoss':tmp[0], 'evalLoss':tmp[1], 'acc':tmp[2]})
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train SeLstm')
    parser.add_argument('--name', type=str, default='KSC',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=3,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    print('*'*5 + 'SeLstm' + '*'*5)
    print('*'*5 + datasetName + '*'*5)
    # res = torch.zeros((len(SAMPLE_PER_CLASS, 3, EPOCHS)))
    for i, num in enumerate(SAMPLE_PER_CLASS):
        print('*' * 5 + 'SAMPLE_PER_CLASS:{}'.format(num) + '*' * 5)
        res = torch.zeros((RUN, 3, EPOCHS))
        for r in range(RUN):
            print('*' * 5 + 'run:{}'.format(r) + '*' * 5)
            ROOT = 'SeLstm/{}/{}/{}'.format(datasetName, num, r)
            if not os.path.isdir(ROOT):
                os.makedirs(ROOT)
            res[r] = main(datasetName, num, r)
        mean = torch.mean(res, dim=0) #[3, EPOCHS]
        viz.line(mean.T, list(range(EPOCHS)), win='SeLstm_SAMPLE_PER_CLASS_{}'.format(num), opts=dict(title='SeLstm_SAMPLE_PER_CLASS_{}'.format(num),
                                                                                               legend=['train loss', 'test loss', 'acc']))
    print('*'*5 + 'FINISH' + '*'*5)