# -*- coding:utf-8 -*-
import random
import numpy as np
import torch.utils.data as Data;
# 把数组写到文件
def dict2pickle(your_dict, out_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(out_file, 'wb') as f:
        # pickle.dump(obj, file[, protocol]):序列化对象，并将结果数据流写入到文件对象中。
        # 参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。
        pickle.dump(your_dict, f)

# 从文件读取数组
def pickle2dict(in_file):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with open(in_file, 'rb') as f:
        # pickle.load(file):反序列化对象。将文件中的数据解析为一个Python对象。
        your_dict = pickle.load(f)
        return your_dict

# 读取文件
def readFileToList(path):
    with open(path,'r',encoding='utf-8') as f:
        content = f.read()
        content = content.replace("\t"," ")
        contents = content.split("\n")
    return contents


def paddingSentence(sentences, max_len):
    newSentences = np.zeros((0, max_len))
    sentence_len = np.zeros((0,))
    for sentence in sentences:
        length = len(sentence)
        sentence_len = np.concatenate((sentence_len, [length]))
        if length > max_len:
            sentence = sentence[:max_len]
            sentence = np.asarray(sentence, dtype=np.int64).reshape(1, -1)
            newSentences = np.concatenate((newSentences, sentence), axis=0)
            # newSentences.append(sentences[:max_len])
        else:
            while len(sentence)<max_len:
                sentence.append(0)
            sentence = np.asarray(sentence, dtype=np.int64).reshape(1, -1)
            # axis=0表示垂直叠加数组，axis=1表示水平叠加数组
            newSentences = np.concatenate((newSentences, sentence), axis=0)
            # newSentences.append(sentence)
    newSentences = newSentences.astype(np.int64)
    sentence_len = sentence_len.astype(np.int64)
    return newSentences, sentence_len


def shuffle(lol, seed=1234567890):
    """
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    lol只有两个元素，第一个是训练数据的集合，第二是训练标签的集合
    这两个集合使用同一个种子打乱顺序，那么打乱以后他们还是一一对应的
    """
    for l in lol:
        # seed() 方法改变随机数生成器的种子，种子改变生成的随机数才改变
        random.seed(seed)
        # shuffle() 方法将序列的所有元素随机排序。返回随机排序后的序列。
        random.shuffle(l)


class MyDataSet(Data.Dataset):

    def __init__(self, data, label, sen_length = None):
        super().__init__()
        self.data = data
        self.label = label
        self.sen_length = sen_length

    def __getitem__(self, index):
        if self.sen_length is None:
            return self.data[index],self.label[index]
        else:
            return self.data[index], self.label[index],self.sen_length[index]

    def __len__(self):
        return len(self.data)