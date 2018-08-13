#*-* encoding=utf-8 *-*
from __future__ import print_function

import copy
import random
import argparse
import numpy as np
import sklearn_crfsuite
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--use_crf', action='store_true', default=False)
parser.add_argument('--model', type=str, choices=['cnn', 'rnn'], default='rnn')
parser.add_argument('--number_of_models', type=int, default=1)
parser.add_argument('--save_path', type=str)	# 训练集
parser.add_argument('--train_set', type=str, default='atis.train.txt')	# 训练集
parser.add_argument('--test_set', type=str, default='atis.test.txt')	# 测试集
parser.add_argument('--output', type=str)	# RNN输出的结果
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--log_every_n_batch', type=int, default=50)
parser.add_argument('--max_step', type=int, default=1000, help='train RNN for this many steps')
parser.add_argument('--crf_step', type=int, default=100, help='train CRF for this many steps')

parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--cell_size', type=int, default=512, help='number of rnn units')
parser.add_argument('--n_channel', type=int, default=512, help='number of cnn channels')
parser.add_argument('--kernel', type=int, default=5, help='kernel size for cnn')
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
    models = []
    if args.model == 'rnn':
        for i in range(args.number_of_models):
            model = Sequence_Tag_RNN_Model(args.embedding_size, args.cell_size, num_words, num_tags)
            models.append(model)
    elif args.model == 'cnn':
        for i in range(args.number_of_models):
            model = Sequence_Tag_CNN_Model(args.embedding_size, args.n_channel, args.kernel, num_words, num_tags)
            models.append(model)
    else:
        raise NotImplementedError

    # train a new model
    if args.train:
        for i,model in enumerate(models):
            train_nn_model(model, train_set, num_tags, i)
            print('finished training {} model.'.format(i))

    # predict results using previously saved model
    if args.test:
        for i,model in enumerate(models):
            save_path = get_save_path(i)
            model.load_state_dict(torch.load(save_path))
            print('restored parameters from {}'.format(save_path))
        test(models, test_set, args.output, id2name_word, id2name_tag)
        compute_atis_fscore(args.test_set, args.output)

    # train a CRF model based on the RNN model
    if args.use_crf:
        train_crf_model(models, train_set, test_set, id2name_tag, num_tags)
        compute_atis_fscore(args.test_set, args.output)


def get_save_path(model_id):
    return args.save_path + '.' + str(model_id)

def compute_atis_fscore(f_true, f_predict):
    true_labels = get_labels_from_output(f_true)
    pred_labels = get_labels_from_output(f_predict)
    m = metrics.precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    print(f_predict + ':\t' + 'precision={}, recall={}, f1_score={}'.format(m[0], m[1], m[2]))
    return m


def get_labels_from_output(result_file):
    print('get labels from {}'.format(result_file))
    labels = []
    f = open(result_file, 'r')
    lines = f.readlines()
    for i,line in enumerate(lines):
        if '\t' not in line:
            print(i, line)
            continue
        _words = line.strip().split('\t')[0].split()
        _labels = line.strip().split('\t')[1].split()
        labels.extend(_labels)
    return labels

def get_logits(model, dataset, id2name_tag, num_tags):
    # 获取RNN输出的结果
    logits = []
    labels = []
    for sample in get_sample_sequentially(dataset):
        np_words, np_tags, np_length = sample
        words = Variable(torch.from_numpy(np_words)).long()
        output = model.forward(words=words,
                               length=np_length)
        output = output.data.numpy()[0] # 形状为(sequence_length, number_of_tags)
        np_tags = np_tags[0]
        np_length = np_length[0]
        logits_per_sentence = []
        labels_per_sentence = []
        for i in range(np_length):
            output_per_word = output[i]
            logits_per_word = dict(('f{}'.format(j), output_per_word[j]) for j in range(num_tags))
            logits_per_sentence.append(logits_per_word)

            label_per_word = np_tags[i]
            labels_per_sentence.append(id2name_tag.get(label_per_word, 'O'))

        logits.append(logits_per_sentence)
        labels.append(labels_per_sentence)
    return logits, labels


def train_nn_model(model, train_set, num_tags, model_id):
    save_path = args.save_path + '.' + str(model_id)
    optimizer = torch.optim.Adam(model.parameters())
    _loss = 0
    for i in range(args.max_step + 1):
        words, tags, length = get_batch_randomly(train_set, args.batch_size)
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

    with open(save_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    print('saved parameters to {}'.format(save_path))


def train_crf_model(models, train_set, test_set, id2name_tag, num_tags):
    # I don't how to ensemble CRF currently so assume there's only one model when using CRF
    # crf.predict() just gives results, not the probabilities
    model = models[0]
    save_path = get_save_path(model_id=0)
    model.load_state_dict(torch.load(save_path))
    # 获取RNN的输出
    logits_train, labels_train = get_logits(model, train_set, id2name_tag, num_tags)
    logits_test, labels_test = get_logits(model, test_set, id2name_tag, num_tags)

    # 利用RNN的输出训练CRF
    crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=args.crf_step)
    crf.fit(logits_train, labels_train)
    y_preds = crf.predict(logits_test)

    # 将CRF结果写入文件
    y_lines = [line.strip().split('\t')[0] for line in open(args.test_set, 'r').readlines()]
    with open(args.output, 'w') as f:
        for y_line, y_pred in zip(y_lines, y_preds):
            f.write(y_line + '\t' + ' '.join(y_pred) + '\n')
    print('tested model, results saved to {}'.format(args.output))


def test(models, test_set, output_file, id2name_word, id2name_tag):
    if type(models) is not list:
        models = [models]
    f = open(output_file, 'w')
    for batch in get_sample_sequentially(test_set):
        np_words, np_tags, np_length = batch
        words = Variable(torch.from_numpy(np_words)).long()
        #tags = Variable(torch.from_numpy(_tags)).long()
        ensemble_output = 0
        for model in models:
            output = model.forward(words=words,
                                   length=np_length)
            ensemble_output += output
        predicted_tags = ensemble_output.max(-1)[1]
        predicted_tags = predicted_tags
        predicted_tags = predicted_tags.data.numpy()
        predicted_tags = predicted_tags[0]
        np_words = np_words[0]

        if  np_words.shape[0] != len(predicted_tags):
            print(np_length)
            print(np_words)
        out_words = ' '.join([id2name_word[w] for w in np_words])
        out_tags = ' '.join([id2name_tag[t] for t in predicted_tags])
        f.write(out_words + '\t' + out_tags + '\n')
    print('tested on {} model(s), results saved to {}'.format(
        args.number_of_models, output_file))


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


def get_batch_randomly(dataset, batch_size):
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


def get_sample_sequentially(dataset):
    num_sample = len(dataset)
    for i in range(num_sample):
        words, tags, length = dataset[i]
        batch_word = np.array([words], np.int32)
        batch_tag = np.array([tags], np.int32)
        batch_length = np.array([length], np.int32)
        yield batch_word, batch_tag, batch_length
    return


class Sequence_Tag_RNN_Model(nn.Module):
    def __init__(self, embedding_size, cell_size, vocabulary_size, num_tags):
        super(Sequence_Tag_RNN_Model, self).__init__()
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

class Sequence_Tag_CNN_Model(nn.Module):
    def __init__(self, embedding_size, n_channel, kernel_size, vocabulary_size, num_tags):
        super(Sequence_Tag_CNN_Model, self).__init__()
        self.embedding_size = embedding_size
        self.n_channel = n_channel
        self.vocab_size = vocabulary_size
        self.num_tags = num_tags
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=PAD_ID)
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_channel,
                                kernel_size=(kernel_size, args.embedding_size),
                                padding=((kernel_size-1)/2, 0))
        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_channel,
                                kernel_size=(kernel_size, n_channel),
                                padding=((kernel_size-1)/2, 0))
        self.output_layer = nn.Linear(self.n_channel, self.num_tags)

    def forward(self, words, length):
        real_batch_size = words.size()[0]
        emb = self.embedding(words)
        emb = emb.view(real_batch_size, 1, -1, args.embedding_size)

        conv_1_out = self.conv_1(emb)
        conv_1_out = conv_1_out.view(real_batch_size, 1, self.n_channel, -1)
        conv_1_out = conv_1_out.transpose(2,3)

        conv_2_out = self.conv_2(conv_1_out)
        conv_2_out = conv_2_out.view(real_batch_size, self.n_channel, -1)
        conv_2_out = conv_2_out.transpose(1,2)

        output = self.output_layer(conv_2_out)
        return output


if __name__ == '__main__':  # 这个不需要改，__main__是Python的一种规定
    main()
