from .read_pgm import read_pgm


def load_map(filename):
    return read_pgm(filename)
