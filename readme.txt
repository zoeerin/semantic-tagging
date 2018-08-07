# for the ATIS RNN model

Train: python main_atis.py --model rnn --save_path checkpoint.rnn.atis --train
Test:  python main_atis.py --model rnn --save_path checkpoint.rnn.atis --test --output result.rnn.atis

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
