import bz2
import pickle
import _pickle as cPickle

def compressed_pickle(title, data):
    with bz2.BZ2File(title + ‘.pbz2’, ‘w’) as f: 
    cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, ‘rb’)
    data = cPickle.load(data)
 return data

