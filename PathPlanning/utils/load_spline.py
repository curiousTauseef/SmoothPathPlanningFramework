import dill
import pickle


def load_spline(pickle_file):
    f = open(pickle_file, 'rb')
    return pickle.load(f)