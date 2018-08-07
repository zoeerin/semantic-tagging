# semantic-tagging
various models for semantic tagging

RNN model:
embedding size=128
number of layer=1
type of layer=RNN
cell size=512
output size=number of tags

commands for training/testing on restaurant test set:
train: python main.py --train
test: python main.py --test

commands for training/testing on movie test set:
train: python main3.py --train
test: python main3.py --test

commands for training/testing on ATIS test set:
python main2.py
