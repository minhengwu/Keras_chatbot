from textblob import TextBlob
from collections import Counter
import pickle
import numpy as np

def get_word_mapping(file_dir, threshold):
    token_count = Counter()
    with open(file_dir) as f:
        data = f.readlines()
        for line in data:
            line= line.replace('|', ' ').lower()
            for i in TextBlob(line).tokens:
                token_count[i] += 1
        f.close()
    token_list = token_count.most_common(n = threshold)
    tokens = ['\t', '\n', '<UNK>', '\s']
    vocab = [i[0] for i in token_list]
    word_to_idx = {}
    idx_to_word = {}
    for i in range(len(tokens)):
        word_to_idx[tokens[i]] = i
        idx_to_word[i] = tokens[i]
    for i in range(len(vocab)):
        word_to_idx[vocab[i]] = i + len(tokens)
        idx_to_word[(i + len(tokens))] = vocab[i]
    return word_to_idx, idx_to_word


def sen_map(sen, len_sen):
    long = len(sen)
    if long > len_sen-2:
        long = len_sen-2
    temp = np.empty(shape = (len_sen),dtype = np.int32)
    temp[0] = 0
    for i in range(long):
        try:
            temp[i+1] = word_to_idx[sen[i]]
        except:
            temp[i+1] = 2
    temp[long+1] = 1
    temp[long+2:25] = 3
    return temp

def generate_data(file_dir, number_of_conv, len_sen):
    X = np.empty(shape = (number_of_conv, len_sen))
    Y = np.empty(shape = (number_of_conv, len_sen))
    with open(file_dir) as f:
        data = f.readlines()
        for i, line in enumerate(data):
            try:
                x,y= line.lower().split('|')
                X[i] = sen_map(TextBlob(x).tokens, len_sen)
                Y[i] = sen_map(TextBlob(y).tokens, len_sen)
            except:
                continue
    return X, Y


if __name__ == '__main__':
    word_to_idx, idx_to_word = get_word_mapping('/home/minheng/nlp_final/dialogues', 5000)
    with open('word_to_idx.pickle', 'wb') as handle:
        pickle.dump(word_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    with open('idx_to_word.pickle', 'wb') as handle:
        pickle.dump(idx_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    X, Y = generate_data('/home/minheng/nlp_final/dialogues', 5000000, 20)
    np.save('X_train', X)
    np.save('Y_train', Y)