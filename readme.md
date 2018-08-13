# ATIS data set

Train 4 rnn models:
```
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 4 --train
```

Test using 1 model:
```
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 1 --output result.rnn.atis --test
```

Test using 4 models ensembled:
```
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 4 --output result.rnn.atis --test
```

If you want to use crf, first train the rnn model with script above, then run:
```
python main_atis.py --save_path ckpt.rnn.atis --model rnn --use_crf --output result.rnn.crf.atis
```
Currently ensembling is not supported for the CRF model
