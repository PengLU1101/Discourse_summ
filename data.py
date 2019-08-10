import pickle, json, os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple, defaultdict

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

path = '/u/lupeng/Project/code/OpenNMT-py/data/cnndm/cnndm'


def get_sent_list(path_r, path_w):
    if os.path.isfile(path_w):
        with open(path_w, 'r') as f:
            doc_list = pickle.load(f)
    else:
        with open(path_r, 'r') as f:
        #line = f.readline()
        #token_txt = sent_tokenize(line)
        #for i, x in enumerate(token_txt):
        #    print(i, "<s>" + x + "<\s>")
        #print(len(token_txt))
        #print(line)
            doc_list = []
            i = 1
            for line in tqdm(f.readlines()):
                token_sents = sent_tokenize(line)
                doc = " ".join([x + "<s>" for x in token_sents])
                doc_list.append(word_tokenize(doc))
                if i == 1:
                    lll = word_tokenize(doc)
                    print(lll)
                    i = 0
                    print(len(lll))
            with open(path_w, 'w') as f_w:
                print("start writing...")
                pickle.dump(doc_list, f_w)
    return doc_list


def get_freq_dict(doc_list):
    freq_dict = defaultdict(int)
    print('Getting freq_dict...')
    for doc in tqdm(doc_list):
        for word in doc:
            freq_dict[word] += 1
    count_freq5 = 0
    count_freq50 = 0
    for x, y in freq_dict.items():
        if y < 5:
            count_freq5 += 1 
        if y < 50:
            count_freq50 += 1
    print("freq < 5", count_freq5/len(freq_dict))
    print("freq < 50", count_freq50/len(freq_dict))
    return freq_dict


    #with open(path+'/train.txt.src_with_sent', 'w') as f_w:
    #    print("start writing...")
    #    f_w.write("\n".join(doc_list))
    #+'/train.txt.src'
def main():
    path = '/u/lupeng/Project/code/OpenNMT-py/data/cnndm/cnndm'
    path_r = path + '/train.txt.src'
    path_w = path + '/train_sent_list'

    l = get_sent_list(path_r, path_w)
    f = get_freq_dict(l)

if __name__ == '__main__':
    main()