#*-* encoding=utf-8 *-*

from __future__ import print_function

import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--train_set', type=str, default='atis.train.txt')	# 训练集
parser.add_argument('--test_set', type=str, default='atis.test.txt')	# 测试集
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--log_every_n_batch', type=int, default=50)
parser.add_argument('--max_step', type=int, default=1000, help='training for this many batches')
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--cell_size', type=int, default=512, help='number of rnn units')
parser.add_argument('--output', type=str, default='output.txt')
args = parser.parse_args()

PAD = '_PAD'
PAD_ID = 0
UNK = '_UNK'
UNK_ID = 1


def main():
    name2id_tag, id2name_tag, name2id_word, id2name_word, num_words, num_tags = get_vocab(args.train_set)
    train_set = read_data_file(args.train_set)
    test_set = read_data_file(args.test_set)
    train_set = convert_data_to_id(train_set, name2id_tag, name2id_word)
    test_set = convert_data_to_id(test_set, name2id_tag, name2id_word)
    model = Sequence_Tag_Model(args.embedding_size, args.cell_size, num_words, num_tags)
    _loss = 0

    # train 先训练
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(args.max_step + 1):
        words, tags, length = get_batch_train(train_set, args.batch_size)
        words = Variable(torch.from_numpy(words)).long()
        tags = Variable(torch.from_numpy(tags)).long()
        output = model.forward(words=words,
                               length=length)
        loss = F.cross_entropy(input=output.view(-1, num_tags),
                               target=tags.view(-1),
                               ignore_index=PAD_ID)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _loss += loss.data[0]
        if i != 0 and i % args.log_every_n_batch == 0:
            print('Batch:', i, '\t', 'Average Loss:', _loss / args.log_every_n_batch)
            _loss = 0

    # test 再测试
    f = open('result.txt', 'w')
    for batch in get_batch_test(test_set):
        np_words, np_tags, np_length = batch
        words = Variable(torch.from_numpy(np_words)).long()
        #tags = Variable(torch.from_numpy(_tags)).long()
        output = model.forward(words=words,
                               length=np_length)
        predicted_tags = output.max(-1)[1]
        predicted_tags = predicted_tags
        predicted_tags = predicted_tags.data.numpy()
        predicted_tags = predicted_tags[0]
        np_words = np_words[0]

        # 输出的文件格式也要改一下
        out_words = ' '.join([id2name_word[w] for w in np_words])
        out_tags = ' '.join([id2name_tag[t] for t in predicted_tags])
        f.write(out_words + '\t' + out_tags + '\n')


def get_vocab(data):
    all_words = set()
    all_tags = set()
    with open(data,'r') as f:
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) >= 2:    # 用for循环比较容易理解
                words, tags = sp
                words = words.split()
                tags = tags.split()
                for w in words:
                    all_words.add(w)
                for t in tags:
                    all_tags.add(t)
    all_words = list(all_words)
    all_tags = list(all_tags)
    all_words.sort()
    all_tags.sort()
    all_tags = [PAD] + [UNK] + all_tags
    all_words = [PAD] + all_words
    name2id_tag = dict([(w,i) for i,w in enumerate(all_tags)]) # maps tag to id
    id2name_tag = dict([(i,w) for i,w in enumerate(all_tags)]) # maps id to tag
    name2id_word = dict([(w,i) for i,w in enumerate(all_words)])   # maps word to id
    id2name_word = dict([(i,w) for i,w in enumerate(all_words)])   # maps id to word

    num_words = len(name2id_word)
    num_tags = len(name2id_tag)
    print('number of unique tags: {}'.format(num_tags))
    print('number of unique words: {}'.format(num_words))
    return name2id_tag, id2name_tag, name2id_word, id2name_word, num_words, num_tags


def read_data_file(data):
    f = open(data, 'r')
    dataset = []
    for line in f.readlines():
        sp = line.strip().split('\t')
        if len(sp) >= 2:
            # 每读一句话加到dataset里
            words, tags = sp
            sentence = words.split()
            sentence_tag = tags.split()
        dataset.append((sentence, sentence_tag, len(sentence)))
    print('number of sentences: {}'.format(len(dataset)))
    return dataset


def convert_data_to_id(dataset, name2id_tag, name2id_word):
    output = []
    for (words,tags,length) in dataset:
        words_ids = [name2id_word.get(word, UNK_ID) for word in words]
        tag_ids = [name2id_tag.get(tag, name2id_tag['O']) for tag in tags]
        output.append((words_ids, tag_ids,length))
    return output


def get_batch_train(dataset, batch_size):
    num_sample = len(dataset)
    select = []
    for i in range(batch_size):
        r = random.randint(0, num_sample-1)
        select.append(copy.deepcopy(dataset[r]))
    max_length = max([_[2] for _ in select])
    for item in select:
        num_pad = max_length - item[2]
        item[0].extend([PAD_ID] * num_pad)
        item[1].extend([PAD_ID] * num_pad)
    select.sort(key=lambda x: x[2], reverse=True)
    batch_word = [_[0] for _ in select]
    batch_tag = [_[1] for _ in select]
    batch_length = [_[2] for _ in select]
    batch_word = np.array(batch_word, np.int32)
    batch_tag = np.array(batch_tag, np.int32)
    batch_length = np.array(batch_length, np.int32)
    return batch_word, batch_tag, batch_length


def get_batch_test(dataset):
    num_sample = len(dataset)
    for i in range(num_sample):
        words, tags, length = dataset[i]
        batch_word = np.array([words], np.int32)
        batch_tag = np.array([tags], np.int32)
        batch_length = np.array([length], np.int32)
        yield batch_word, batch_tag, batch_length
    return


class Sequence_Tag_Model(nn.Module):
    def __init__(self, embedding_size, cell_size, vocabulary_size, num_tags):
        super(Sequence_Tag_Model, self).__init__()
        self.embedding_size = embedding_size
        self.cell_size = cell_size
        self.vocab_size = vocabulary_size
        self.num_tags = num_tags
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=PAD_ID)
        self.rnn = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.cell_size,
                          bidirectional=True,
                          batch_first=True)
        self.output_layer = nn.Linear(self.cell_size * 2, self.num_tags)

    def forward(self, words, length):

        emb = self.embedding(words)
        emb = nn.utils.rnn.pack_padded_sequence(emb, length, batch_first=True)
        output, _ = self.rnn.forward(emb)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.output_layer(output)
        return output


if __name__ == '__main__':  # 这个不需要改，__main__是Python的一种规定
    main()
