# -*- coding:utf-8 -*-
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as Data
import preProcess
import utils
import time
from torch.autograd import Variable
from model.CNN_LSTM import CNN_LSTM
from model.blstm import BLSTM
from model.lstm import LSTM
from model.cnn import CNN
from utils import MyDataSet

BATCH_SIZE = 8
EPOCH = 100
max_len = 40
l_rate = 0.5
ReTrain = True
Model = "CNN_LSTM"


# 开始运行程序
def startProgress():
    ret = preProcess.getEmbedding()
    embeddings = ret["embdeddings"]
    word2id = ret["word2id"]
    id2word = ret["id2word"]
    assert len(embeddings) == len(word2id)
    assert len(word2id) == len(id2word)
    data = preProcess.getData()
    trainData = data[0]
    testData = data[1]
    trainSentence = trainData[0]
    trainLabel = trainData[1]
    testSentence = testData[0]
    testLabel = testData[1]
    # print('trainSentence：',trainSentence,
    #       '\ntrainLabel：',trainLabel,
    #       '\ntestSentence：',testSentence,
    #       '\ntestLabel：',testLabel
    #       ,"\ntrain数据个数:",len(trainSentence)
    #       ,"\ntest数据个数：",len(testSentence))
    assert len(trainSentence) == len(trainLabel)
    assert len(testSentence) == len(testLabel)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = torch.from_numpy(embeddings)

    trainSentence, train_sentence_length = utils.paddingSentence(trainSentence, 40)
    testSentence, test_sentence_length = utils.paddingSentence(testSentence, 40)
    train_sentence_length = torch.LongTensor(train_sentence_length).cuda()
    test_sentence_length = torch.LongTensor(test_sentence_length).cuda()

    trainSentence = torch.LongTensor(trainSentence)
    trainLabel = torch.LongTensor(trainLabel)
    trainSentence = Variable(trainSentence)
    trainLabel = Variable(trainLabel)
    trainSentence = trainSentence.cuda()
    trainLabel = trainLabel.cuda()

    testSentence = torch.LongTensor(testSentence)
    testLabel = torch.LongTensor(testLabel)
    testSentence = Variable(testSentence)
    testLabel = Variable(testLabel)
    testSentence = testSentence.cuda()
    testLabel = testLabel.cuda()

    loader, test_loader, model = prepare(embeddings,
                                         trainSentence, trainLabel, train_sentence_length, testSentence, testLabel,
                                         test_sentence_length)

    if ReTrain:
        # trainCNN(embdeddings, trainSentence, trainLabel, testSentence, testLabel)
        train(loader, model, test_loader)
    test(test_loader)


# 准备数据
def prepare(embeddings, trainSentence, trainLabel, train_sentence_length, testSentence, testLabel,
            test_sentence_length):
    dataset = MyDataSet(trainSentence, trainLabel, train_sentence_length)
    loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = MyDataSet(testSentence, testLabel, test_sentence_length)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if Model == 'BLSTM':
        model = BLSTM(embeddings, input_dim=100, hidden_dim=50, num_layers=2, output_dim=2, max_len=40, dropout=0.5)
    elif Model == 'LSTM':
        model = LSTM(embeddings, input_dim=100, hidden_dim=50, num_layers=2, output_dim=2, max_len=40, dropout=0.5)
    elif Model == 'CNN':
        model = CNN(embeddings, 100)
    elif Model == 'CNN_LSTM':
        model = CNN_LSTM(embeddings, sequence_length=40, vocab_size=embeddings.size(0), embedding_size=embeddings.size(1),
                         filter_sizes=[1, 2, 3],
                         num_filters=32, num_classes=2, dropout=0.5, num_hidden=100, num_layers=1)
    model.cuda()
    return loader, test_loader, model


def train(loader, model, test_loader):
    # bestScore = test(test_loader)
    bestScore = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
    '''
       class torch.nn.CrossEntropyLoss(weight=None, size_average=True)[source]
       调用时参数：
       input : 包含每个类的得分，2-D tensor,shape为 batch*n
       target: 大小为 n 的 1—D tensor，包含类别的索引(0到 n-1)。
       这个交叉熵函数非常神奇，他要求的input即各个类的预测结果是类似于softmax的形式，表示每个类的得分，是一个2维的张量，即一个表格
       而target即真实类别并非我们想当然的one-hot形式，而是一个一维的张量，即一个向量，向量中的每个数代表该输入的真实类别编号
       所以我们的input（即模型的output）可能是[[0.434，0.754],[0.843,0.211]……]，而我们的target是[0,1……]
    '''
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        print("开始新一轮训练：epoch = %d, 总训练轮数：%d" % (epoch + 1, EPOCH))
        tic = time.time()
        for step, (sentence, label, sentence_length) in enumerate(loader):
            model.zero_grad()
            output = model(sentence, sentence_length)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            # print("损失：%.5f" % loss)
        print("本轮训练结束，用时%.5f秒。,损失：%.5f" % ((time.time() - tic), loss))
        score = test(test_loader, model)
        print("本次正确率：%f, 历史最高正确率：%f" % (score, bestScore))
        if score > bestScore:
            bestScore = score
            print("最高正确率更新为 %f" % score)
            torch.save(model, Model + 'Best.pkl')  # save entire net 保存整个神经网络，参数和网络结构都保存
        print('\n')
    print("训练完成\n")

    # torch.save(model, Model + ".pkl")


def test(loader, model=None):
    count = 0
    num = len(loader.dataset)
    if model is None:
        cnn = torch.load(Model + "Best.pkl")
        print("测试" + Model + "模型:")
    else:
        cnn = model
    cnn.cuda()
    for (sentence, label, sentence_length) in loader:
        output = cnn(sentence, sentence_length)
        _, pred = torch.max(output, dim=1)
        for i in range(len(label)):
            if pred[i] == label[i]:
                count = count + 1
    print("count:", count, "\tnum:", num, '\t正确率', count / num)
    return count / num


if __name__ == '__main__':
    startProgress()
