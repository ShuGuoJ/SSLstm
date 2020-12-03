import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat, savemat
from utils import loadLabel
from HSIDataset import HSIDatasetV1, DatasetInfo
from torch.utils.data import DataLoader
import os
from torch import nn
from Model.module import SeLstm
import argparse
from visdom import Visdom

isExists = lambda path: os.path.exists(path)
EPOCHS = 10
LR = 1e-1
BATCHSZ = 10
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
viz = Visdom()


def train(model, criterion, optimizer, dataLoader, DEVICE, mode='spectra'):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(DEVICE)
    trainLoss = []
    for step, ((spectra, neighbor_region), target) in enumerate(dataLoader):
        # spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        # neighbor_region = neighbor_region.permute((0, 3, 1, 2))
        target = target.to(DEVICE)
        input = spectra.to(DEVICE) if mode == 'spectra' else neighbor_region.to(DEVICE)
        out = model(input)

        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 1e-1)
        optimizer.step()

        if step%5 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} loss:{} lr:{}'.format(step, loss.item(), lr))
    return model, float(np.mean(trainLoss))


def test(model, criterion, dataLoader, DEVICE, mode='spectra'):
    model.eval()
    evalLoss, correct = [], 0
    for (spectra, neighbor_region), target in dataLoader:
        # spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        # neighbor_region = neighbor_region.permute((0, 3, 1, 2))
        target = target.to(DEVICE)
        input = spectra.to(DEVICE) if mode == 'spectra' else neighbor_region.to(DEVICE)
        logits = model(input)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)


def main(datasetName, n_sample_per_class, run):
    # 加载数据和标签
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    info = DatasetInfo.info[datasetName]
    data_path = "./data/{0}/{0}.mat".format(datasetName)
    label_path = './trainTestSplit/{}/sample{}_run{}.mat'.format(datasetName, n_sample_per_class, run)
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]
    bands = data.shape[2]
    isExists(label_path)
    trainLabel, testLabel = loadLabel(label_path)
    # 数据转换
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(np.int), testLabel.astype(np.int)
    nc = int(np.max(trainLabel))
    trainDataset = HSIDatasetV1(data, trainLabel, patchsz=info['patchsz'])
    testDataset = HSIDatasetV1(data, testLabel, patchsz=info['patchsz'])
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)
    model = SeLstm(1, info['hzOfSe'], nc)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    res = torch.zeros((3, EPOCHS))
    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader, DEVICE=DEVICE)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader, DEVICE=DEVICE)
        viz.line([[trainLoss, evalLoss]], [epoch], win='train&test loss', update='append')
        viz.line([acc], [epoch], win='accuracy', update='append')
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4%}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        res[0, epoch], res[1, epoch], res[2, epoch] = trainLoss, evalLoss, acc
        if epoch%5 == 0:
            torch.save(model.state_dict(), os.path.join(
                'SeLstm/{0}/{1}/{2}/SeLstm_sample{1}_run{2}_epoch{3}.pkl'.format(datasetName, n_sample_per_class,
                                                                                 run, epoch)))
        scheduler.step()
    res = res.numpy()
    savemat('SeLstm/{0}/{1}/{2}/res.mat'.format(datasetName, n_sample_per_class, run), {'trainLoss':res[0], 'evalLoss':res[1], 'acc':res[2]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train SeLstm')
    parser.add_argument('--name', type=str, default='KSC',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    # main(datasetName, n_sample_per_class, run, encoderPath=None)
    viz.line([[0., 0.]], [0.], win='train&test loss', opts=dict(title='train&test loss',
                                                                legend=['train_loss', 'test_loss']))
    viz.line([0.,], [0.,], win='accuracy', opts=dict(title='accuracy',
                                                     legend=['accuracy']))
    main(datasetName, 100, 8)

