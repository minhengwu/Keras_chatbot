from create_model import build_model
from keras.utils.np_utils import to_categorical
import numpy as np

def to_cat(data):
    return np.array([to_categorical(i, 5004) for i in data])

def seq_seq(model, epoch, X, Y):
    for _ in range(epoch):
        randomizer = np.random.randint(0, 5000000, 10000)
        x_train = to_cat(X[randomizer])
        y = to_cat(Y[randomizer])
        y_train = y[:, :-1, :]
        y_test = y[:, 1:, :]
        model.fit([x_train, y_train], y_test,
                  epochs=1,
                  validation_split=0.2)
    model.save_weights('my_model_weights.h5')


if __name__ == '__main__':
    num_encoder_tokens = 5004
    num_decoder_tokens = 5004
    latent_dim = 256

    model, _ ,_ = build_model(latent_dim, num_encoder_tokens, num_decoder_tokens)
    X = np.load('X_train.npy')
    Y = np.load('X_train.npy')
    seq_seq(model, 200,X,Y)