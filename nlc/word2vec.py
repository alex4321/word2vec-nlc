from gensim.models import Word2Vec as Word2VecModel
import math


class Vector(dict):
    def __setitem__(self, key, value):
        """
        Set vector item
        :param key: word
        :type key: str
        :param value: similarity
        :type value: float|int
        """
        assert isinstance(key, str)
        assert isinstance(value, float) or isinstance(value, int)
        super(Vector, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get vector item
        :param item: word
        :type item: str
        :return: similarity
        :rtype: float|int
        """
        assert isinstance(item, str)
        return super(Vector, self).__getitem__(item)

    def _add_k(self, other, kooficient):
        """
        Add other vector with kooficient
        :param other: Other vector
        :type other: Vector
        :param kooficient: kooficient
        :type kooficient: float|int
        :return: new vector
        :rtype: Vector
        """
        result = Vector(self)
        for key in other.keys():
            if key not in result.keys():
                result[key] = 0
            result[key] = kooficient * other[key]
        return result

    def __add__(self, other):
        """
        Add other vector
        :param other: Other vector
        :type other: Vector
        :param kooficient: kooficient
        :type kooficient: float
        :return: new vector
        :rtype: Vector
        """
        return self._add_k(other, 1)

    def __sub__(self, other):
        """
        Sub other vector
        :param other: Other vector
        :type other: Vector
        :param kooficient: kooficient
        :type kooficient: float
        :return: new vector
        :rtype: Vector
        """
        return self._add_k(other, -1)

    def length(self):
        """
        Get vector angle
        :return: angle
        :rtype: float
        """
        result = 0
        for key in self.keys():
            result += self[key] * self[key]
        return math.sqrt(result)


class Word2Vec:
    __models = {}

    def __init__(self, path):
        """
        Initialize new word2vec instance
        :param path: ipath to model binary file
        :type path: str
        """
        self.model = Word2Vec.__models[path]

    def __new__(cls, path):
        """
        Initialize new word2vec instance
        :param path: ipath to model binary file
        :type path: str
        """
        if path not in Word2Vec.__models.keys():
            print("Loading model", path)
            Word2Vec.__models[path] = Word2VecModel.load_word2vec_format(path, binary=True)
            print("Model loaded")
        return super(Word2Vec, cls).__new__(cls)

    def _filter_words(self, words):
        result = []
        for word in words:
            try:
                self.model[word]
                result.append(word)
            except KeyError:
                pass
        return result

    def similar(self, positive, negative=None):
        """
        Get vector
        :param positive: words
        :type positive: list[str]
        :param negative: words
        :type negative: list[str]
        :return: dictionary of "word"-"similarity"
        :rtype: Vector
        """
        if negative is None:
            negative = []
        positive_words = self._filter_words(positive)
        negative_words = self._filter_words(negative)
        items = self.model.most_similar(positive=positive_words, negative=negative_words)
        result = {}
        for item in items:
            word, similarity = item
            result[word] = similarity
        return Vector(result)
