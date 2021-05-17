'''

一些NLP数据/语料下载
https://blog.csdn.net/u011736505/article/details/73292678
'''
from pathlib import Path
import platform

if 'hamming' in platform.node():
    from tensorflow.python.keras import regularizers, initializers, optimizers, callbacks
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    from tensorflow.python.keras.preprocessing.text import Tokenizer

import re
from tqdm import tqdm_notebook
if 'hamming' in platform.node():
    from tqdm import tqdm as tqdm_notebook

from nltk.corpus import stopwords

from tensorflow.python.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "glove/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

train = pd.read_csv('data/toxic/train.csv')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = train[labels].values
comments_train = train['comment_text']
comments_train = list(comments_train)

# Text Pre-processing
def clean_text(text, remove_stopwords = True):
    output = ""
    text = str(text).replace("\n", "")
    text = re.sub(r'[^\w\s]','',text).lower()
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
    else:
        output = text
    return str(output.strip())[1:-3].replace("  ", " ")

texts = []

for line in tqdm_notebook(comments_train, total=159571):
    texts.append(clean_text(line))

print('Sample data:', texts[1], y[1])


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))