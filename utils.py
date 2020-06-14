import bz2
import pickle
import _pickle as cPickle

def compressed_pickle(file_path, data):
    with bz2.BZ2File(file_path + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(file_path):
    data = bz2.BZ2File(file_path+ '.pbz2', 'rb')
    data = cPickle.load(data)
    return data

