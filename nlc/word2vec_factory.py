from .word2vec_reader import Word2VecReader

_readers = {}


def load_word2vec_reader(path):
    """
    Get reader
    :param path: path
    :type path: str
    :return: reader
    :rtype: Word2VecReader
    """
    if path not in _readers.keys():
        _readers[path] = Word2VecReader(path)
    return _readers[path]
