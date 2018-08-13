from __future__ import print_function
import json
import copy
import random
import argparse
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F

PAD = '_PAD'
UNK = '_UNK'
PAD_ID = 0
UNK_ID = 1

def read_file(file_name):
    foods = json.load(open(file_name))

    data_set = []

    for food_id in foods:
        attr_names = foods[food_id].keys()
        attr_names.remove(u'usda_name')
        attr_names = [u'usda_name'] + attr_names
        attr_names = [str(_) for _ in attr_names]
        break

    for food_id in foods:
        attrs = foods[food_id]
        sample_usda_name = str(attrs[attr_names[0]]).split()
        sample_attrs = []
        for attr_name in attr_names[1:]:
            attr_value = attrs[attr_name]
            attr_value = str(attr_value) if attr_value is not None else 'null'
            sample_attrs.append(attr_value)
        data_set.append([sample_usda_name, sample_attrs, len(sample_usda_name)])
    return data_set, attr_names[1:]


def build_name_vocab(data_set):
    vocab_usda_name = {}
    vocab_usda_name[PAD] = PAD_ID
    vocab_usda_name[UNK] = UNK_ID
    count = 2
    for item in data_set:
        names = item[0]
        for w in names:
            if w not in vocab_usda_name:
                vocab_usda_name[w] = count
                count += 1
    return vocab_usda_name


def build_attrs_vocab(data_set, i):
    vocab = {}
    rev_vocab = {}
    vocab['null'] = 0
    rev_vocab[0] = 'null'
    count = 1
    for item in data_set:
        attr = item[1][i]
        if attr not in vocab:
            vocab[attr] = count
            rev_vocab[count] = attr
            count += 1
    return vocab, rev_vocab


def convert_data_to_id(dataset, vocab_word, vocab_attrs):
    output = []
    for (words,attrs,length) in dataset:
        words_ids = [vocab_word.get(word, UNK_ID) for word in words]
        attrs_ids = [vocab.get(attr, vocab['null']) for attr, vocab in zip(attrs, vocab_attrs)]
        output.append((words_ids, attrs_ids,length))
    return output


class Food_Model(nn.Module):
    def __init__(self, embedding_size, cell_size, vocabulary_size, num_tags):
        super(Food_Model, self).__init__()
        self.embedding_size = embedding_size
        self.cell_size = cell_size
        self.vocab_size = vocabulary_size
        self.num_tags = num_tags
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=PAD_ID)
        self.rnn = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.cell_size,
                          bidirectional=True,
                          batch_first=True)
        self.output_layers = [nn.Linear(self.cell_size * 2, n_attr) for n_attr in num_tags]


    def forward(self, words, length):
        emb = self.embedding(words)
        # emb = nn.utils.rnn.pack_padded_sequence(emb, length, batch_first=True)
        output, _ = self.rnn.forward(emb)
        # output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = [self.output_layers[i](output[:,0]) for i in range(len(self.num_tags))]
        return logits


def train(model, train_set, num_tags):
    optimizer = torch.optim.Adam(model.parameters())
    _loss = 0
    for i in range(5000):
        words, tags12, length = get_batch_randomly(train_set, 50)
        words = Variable(torch.from_numpy(words)).long()
        tags12 = [Variable(torch.from_numpy(t)).long() for t in tags12]
        logits = model.forward(words=words,
                               length=length)
        loss = [F.cross_entropy(input=logits[j].view(-1, num_tags[j]),
                                target=tags12[j].view(-1))
                for j in range(len(num_tags))]
        loss = sum(loss) / 12
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _loss += loss.data[0]
        if i != 0 and i % 50 == 0:
            print('Batch:', i, '\t', 'Average Loss:', _loss / 50)
            _loss = 0

        if i != 0 and i % 1000 == 0:
            with open('ckpt.food', 'wb') as f:
                torch.save(food_model.state_dict(), f)


def test(model, test_set, num_tags, attr_names, rev_vocab_attrs):
    for i in range(10):
        words, tags12, length = get_batch_randomly(test_set, 1, i)
        words = Variable(torch.from_numpy(words)).long()
        # tags12 = [Variable(torch.from_numpy(t)).long() for t in tags12]
        logits = model.forward(words=words,
                               length=length)
        for logit, vocab, attr_name in zip(logits, rev_vocab_attrs, attr_names):
            print(attr_name, '\t', [vocab[_] for _ in logit.max(-1)[1].data][0])


def simple_test(model, word_vocab, attr_names, rev_vocab_attrs):
    usda_names = ['Yogurt, Greek, strawberry, nonfat',
                  'Wheat flour, white (industrial), 13% protein, bleached, unenriched',
                  'Waffles, gluten-free, frozen, ready-to-heat',
                  'Turkey, whole, back, meat only, cooked, roasted',
                  'Tomato powder',
                  'Sweet potato, canned, vacuum pack',
                  'Sugars, maple',
                  'Strawberries, raw',
                  'SUBWAY, cold cut sub on white bread with lettuce and tomato',
                  'Spices, curry powder',
                  'Soup, vegetarian vegetable, canned, prepared with equal volume water',
                  'Sour cream, reduced fat',
                  'Soursop, raw',
                  'Snacks, rice cakes, brown rice, multigrain, unsalted',
                  'Sausage, turkey, fresh, cooked']
    for usda_name in usda_names:
        usda_name = usda_name.lower()
        # print(vocab.get('rice'))
        words = [[word_vocab.get(w, UNK_ID) for w in usda_name.split()]]
        # print(words)
        words = np.array(words)
        length = [len(words[0])]
        words = Variable(torch.from_numpy(words)).long()
        logits = model.forward(words=words, length=length)
        print(usda_name)
        # print(words)
        # print(length)
        # print(logits[1])
        for logit, vocab, attr_name in zip(logits, rev_vocab_attrs, attr_names):
            print(attr_name, '\t', [vocab[_] for _ in logit.max(-1)[1].data][0])
        print()


def get_batch_randomly(dataset, batch_size, index=None):
    num_sample = len(dataset)
    select = []
    for i in range(batch_size):
        if index is None:
            r = random.randint(0, num_sample-1)
        else:
            r = index
        select.append(copy.deepcopy(dataset[r]))
    max_length = max([_[2] for _ in select])
    for item in select:
        num_pad = max_length - item[2]
        item[0].extend([PAD_ID] * num_pad)
        # item[1].extend([PAD_ID] * num_pad)
    select.sort(key=lambda x: x[2], reverse=True)
    batch_word = [_[0] for _ in select]
    batch_tag = [[_[1][i] for _ in select] for i in range(12)]
    batch_length = [_[2] for _ in select]
    batch_word = np.array(batch_word, np.int32)
    batch_tag = np.array(batch_tag, np.int32)
    batch_length = np.array(batch_length, np.int32)
    return batch_word, batch_tag, batch_length

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    data_set, attr_names = read_file('all-foods-json.lower')

    vocab_word = build_name_vocab(data_set)
    vocab_attrs = []
    rev_vocab_attrs = []
    for i in range(12):
        vocab, rev_vocab = build_attrs_vocab(data_set, i)
        vocab_attrs.append(vocab)
        rev_vocab_attrs.append(rev_vocab)

    data_set = convert_data_to_id(data_set, vocab_word, vocab_attrs)

    number_words = len(vocab_word)
    number_attrs = [len(_) for _ in vocab_attrs]

    food_model = Food_Model(128, 256, number_words, number_attrs)

    train(food_model, data_set, number_attrs)
    # test(food_model, data_set, number_attrs, attr_names, rev_vocab_attrs)

    # if args.test:
    # food_model.load_state_dict(torch.load('ckpt.food'))
    simple_test(food_model, vocab_word, attr_names, rev_vocab_attrs)
