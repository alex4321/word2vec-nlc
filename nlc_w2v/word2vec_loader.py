from gensim.models import Word2Vec

loaded = {}


def load(path):
    """
    Load word2vec binary model or give loaded
    :param path: path
    :type path: str
    :return: model
    :rtype: Word2Vec
    """
    if path in loaded.keys():
        return loaded[path]
    else:
        result = Word2Vec.load_word2vec_format(path, binary=True)
        loaded[path] = result
        return result


def get_path(model):
    """
    Get path of word2vec model
    :param model: model
    :type model: Word2Vec
    :return: path
    :rtype: str
    """
    for path, instance in loaded.items():
        if hash(instance) == hash(model):
            return path
    return ''
