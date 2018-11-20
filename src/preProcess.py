# -*- coding:utf-8 -*-
import utils
import numpy as np

trainDataPath = "../data/mr/MR.task.train.sentences"
testDataPath = "../data/mr/MR.task.test.sentences"
trainLabelPath = "../data/mr/MR.task.train.labels"
testLabelPath = "../data/mr/MR.task.test.labels"
embeddingsPath = "../data/glove.6B.100d/glove.6B.100d.txt"

#得到词嵌入
def getEmbedding():
    ret = dict()
    embdeddings = utils.pickle2dict("../data/mr/embeddings_glove.pkl")
    word2id = utils.pickle2dict("../data/mr/word2id_glove.pkl")
    id2word = utils.pickle2dict("../data/mr/id2word_glove.pkl")
    ret["embdeddings"] = embdeddings
    ret["word2id"] = word2id
    ret["id2word"] = id2word
    return ret

def getData():
    return utils.pickle2dict("../data/mr/data.pkl")

#读取词嵌入文件专用方法
def readEmbeddingFile(path):
    embdeddings = []
    word2id = dict()
    word2id["_padding"] = 0  # PyTorch Embedding lookup need padding to be zero
    word2id["_unk"] = 1
    i = 0
    with open(path,"r",encoding='utf-8') as f:
        for line in f:
            line = line.replace("\t"," ")
            es = line.split(" ")
            word = es[0]
            emb = [float(x) for x in es[1:]]
            embdeddings.append(emb)
            word2id[word] = len(word2id)
    length = len(embdeddings[0])
    embdeddings.insert(0,np.zeros(length))
    embdeddings.insert(1,np.ones(length))
    embdeddings = np.asarray(embdeddings,dtype=np.float32)
    # 下面这句代码其实不用写
    embdeddings = embdeddings.reshape(len(embdeddings),length)
    id2word = dict((word2id[word],word) for word in word2id)
    utils.dict2pickle(embdeddings,"../data/mr/embeddings_glove.pkl")
    utils.dict2pickle(word2id,"../data/mr/word2id_glove.pkl")
    utils.dict2pickle(id2word,"../data/mr/id2word_glove.pkl")

    return embdeddings,word2id,id2word

#读取数据
def readData():
    print("readData开始执行\n")
    data = []
    trainData = []
    testData = []
    trainDataList = utils.readFileToList(trainDataPath)
    testDataList = utils.readFileToList(testDataPath)
    trainLabelList = utils.readFileToList(trainLabelPath)
    testLabelList = utils.readFileToList(testLabelPath)

    trainLabelList = [int(x) for x in trainLabelList if x != '']
    testLabelList = [int(x) for x in testLabelList if x !='']
    trainLabel = np.asarray(trainLabelList,dtype=np.int64).reshape(len(trainLabelList))
    testLabel = np.asarray(testLabelList,dtype=np.int64).reshape(len(testLabelList))

    ret = getEmbedding()
    word2id = ret["word2id"]
    print(word2id)
    print("开始转换成id\n")
    for data in trainDataList:
        print("train本次转换：",data,"\n")
        words = data.split(" ")
        if data!='':
            l = lambda word: word2id[word] if word in word2id else word2id["_unk"]
            words = [l(word) for word in words]
            print(words)
        else:
            continue
        trainData.append(words)
        print("len(trainData)",len(trainData))
    for data in testDataList:
        print("test本次转换：", data, "\n")
        words = data.split(" ")
        if data!='':
            l = lambda word:word2id[word] if word in word2id else word2id["_unk"]
            words = [l(word) for word in words]
        else:
            continue
        testData.append(words)

    trainData = [trainData,trainLabel]
    testData = [testData,testLabel]

    print("转换成id结束\n", "len(trainData[0])", len(trainData[0]), "len(trainData[1])", len(trainData[1]))
    assert len(trainData[0]) == len(trainData[1])
    assert len(testData[0]) == len(testData[1])
    data = [trainData,testData]
    utils.shuffle(data[0], seed=123456)
    utils.shuffle(data[1],seed=123456)
    utils.dict2pickle(data,"../data/mr/data.pkl")


def process():
    # readEmbeddingFile(embeddingsPath)
    readData()


if __name__=='__main__':
    process()