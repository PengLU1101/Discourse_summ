import pickle, json, os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple, defaultdict, OrderedDict

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

path = '/u/lupeng/Project/code/OpenNMT-py/data/cnndm/cnndm'

def get_tokenized_sent(sent_lists):
    if len(sent_lists):
        return [["<sent>"] + word_tokenize(x) + ["</sent>"] for x in sent_lists]
    else:
        return []


def build_sentlist(file_dict, vocab=None, word2id=None):
    if (not word2id) and (not vocab):
        word2id = OrderedDict([('<pad>', 0), ('<unk>', 1), ('<sent>', 2), ('</sent>', 3)])
        vocab = defaultdict(int)
    src_token, tgt_token = [], []

    with open(os.path.join(path, file_dict['src']), 'r') as f_src, open(os.path.join(path, file_dict['tgt']),
                                                                        'r') as f_tgt:
        for src, tgt in tqdm_notebook(zip(f_src, f_tgt)):
            tokened_src_list = get_tokenized_sent(sent_tokenize(src.lower().strip()))
            tokened_tgt_list = get_tokenized_sent(sent_tokenize(tgt.lower().strip()))

            for word_list in tokened_src_list:
                for word in word_list:
                    vocab[word] += 1
                    if word not in word2id:
                        word2id[word] = len(word2id)
            for word_list in tokened_tgt_list:
                for word in word_list:
                    vocab[word] += 1
                    if word not in word2id:
                        word2id[word] = len(word2id)

            if len(tokened_src_list) and len(tokened_src_list[0]) and len(tokened_tgt_list) and len(
                    tokened_tgt_list[0]):
                src_token.append(tokened_src_list)
                tgt_token.append(tokened_tgt_list)
    return vocab, word2id, src_token, tgt_token

def combine_list_dict(src, tgt):
    return  {'src': src, 'tgt': tgt}

def build_pkl(files_dict, path, save_dict):
    path_vocab = os.path.join(path, save_dict['vocab'])
    if os.path.isfile(path_vocab):
        vocab = read_pkl(path_vocab)
        #with open(path_vocab, 'rb') as f:
        #    vocab = pickle.load(f)
    else:
        vocab = {}
    path_data = os.path.join(path, save_dict['data'])
    if os.path.isfile(path_data):
        data = read_pkl(path_data)
        #with open(path_data, 'rb') as f:
        #    data = pickle.load(f)
    else:
        data = {}
    if vocab and data:
        print("Vocabulary and processed files exist.")
        print("Vocab file size %d M" %(os.path.getsize(path_vocab) >> 20))
        print("Processed file size %d M" % (os.path.getsize(path_data) >> 20))

    else:
        vocab, word2id, test_src_token, test_tgt_token = build_pkl(test_file_dict)
        vocab, word2id, val_src_token, val_tgt_token = build_pkl(val_file_dict, vocab, word2id)
        vocab, word2id, train_src_token, train_tgt_token = build_pkl(train_file_dict, vocab, word2id)

        data = {'word2id': word2id,
                'train_token': combine_list_dict(train_src_token, train_tgt_token),
                'val_token': combine_list_dict(val_src_token, val_tgt_token),
                'test_token': combine_list_dict(test_src_token, test_tgt_token)
               }




def read_pkl(path):
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict

def save_pkl(path, file):
    with open(path, 'wb+') as f:
        pickle.dump(file, f)





def main():
    path = '/u/lupeng/Project/code/OpenNMT-py/data/cnndm/cnndm'
    path_r = path + '/train.txt.src'
    path_w = path + '/train_sent_list'

    l = get_sent_list(path_r, path_w)
    f = get_freq_dict(l)

if __name__ == '__main__':
    main()