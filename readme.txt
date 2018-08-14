
#Train 4 rnn models on ATIS:
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 4 --train

#Test using 4 models ensembled:
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 4 --output result.rnn.atis --test

#Test using 1 RNN model:
python main_atis.py --save_path ckpt.rnn.atis --model rnn --number_of_models 1 --output result.rnn.atis --test

#If you want to use crf, first train the rnn model with script above, then run:
python main_atis.py --save_path ckpt.rnn.atis --model rnn --use_crf --output result.rnn.crf.atis
Currently ensembling is not supported for the CRF model

# for the ATIS CNN model

Train: python main_atis.py --model cnn --save_path checkpoint.cnn.atis --train 
Test:  python main_atis.py --model cnn --save_path checkpoint.cnn.atis --test --output result.cnn.atis


# for the RESTAURANT RNN model

Train: python main_restaurant.py --model rnn --save_path checkpoint.rnn.restaurant --train
Test:  python main_restaurant.py --model rnn --save_path checkpoint.rnn.restaurant --test --output result.rnn.restaurant

# for the RESTAURANT CNN model

Train: python main_restaurant.py --model cnn --save_path checkpoint.cnn.restaurant --train 
Test:  python main_restaurant.py --model cnn --save_path checkpoint.cnn.restaurant --test --output result.cnn.restaurant 

# for the MOVIE RNN model

Train: python main_movie.py --model rnn --save_path checkpoint.rnn.movie --train
Test:  python main_movie.py --model rnn --save_path checkpoint.rnn.movie --test --output result.rnn.movie

# for the MOVIE CNN model

Train: python main_movie.py --model cnn --save_path checkpoint.cnn.movie --train 
Test:  python main_movie.py --model cnn --save_path checkpoint.cnn.movie --test --output result.cnn.movie 

# for the usda_food model
python food_label.py
the model will automatically test after 5000 steps of training
