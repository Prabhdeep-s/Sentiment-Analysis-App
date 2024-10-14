from gensim.models import KeyedVectors
import numpy as np
import sys
sys.path.append('./')

model = KeyedVectors.load(r'..\WordEmbeddings\Word2Vec\w2vModel.kv')

# the corpus argument is a list of sentences made up of comma-seperated tokens
def getWord2VecVectorsPreTrained(corpus):
    def word_vector(sentence):
        wv_res = np.zeros(model.vector_size)
        ctr = 1
        for w in sentence:
            if w in model:
                ctr += 1
                wv_res += model[w]
        wv_res = wv_res/ctr
        return wv_res
    vectors = [word_vector(text) for text in corpus]
    return vectors