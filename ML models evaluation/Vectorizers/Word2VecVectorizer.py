from gensim.models import Word2Vec
import numpy as np

# the corpus argument is a list of - sentences made up of continuous tokens
def getWord2VecVectors(corpus, sizeOfVectors):
    model = Word2Vec(corpus, min_count=1, vector_size=sizeOfVectors, window=5, workers=4)
    def word_vector(sentence):
        wv_res = np.zeros(sizeOfVectors)
        ctr = 1
        for w in sentence:
            if w in model.wv:
                ctr += 1
                wv_res += model.wv[w]
        wv_res = wv_res/ctr
        return wv_res
    vectors = [word_vector(text) for text in corpus]
    return vectors