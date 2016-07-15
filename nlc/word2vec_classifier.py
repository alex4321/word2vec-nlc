import math
import numpy
from .classes import Classes
from .classifier import Classifier
from .exploder import Exploder
from .word2vec_factory import load_word2vec_reader
from .word2vec_classifier_network import Word2VecClassifierNetwork


class Word2VecClassifier(Classifier):
    """
    Classifier based on word2vec-based distances
    """

    def __init__(self, path=None, stop_words=None, config=None):
        """
        Initialize NLC
        :param path: path to word2vec binary model
        :type path: str|None
        :param stop_words: stop words
        :type stop_words: list[str]|None
        :param config: configuration dict
        :type config: dict
        """
        assert (path is not None and stop_words is not None) ^ (config is not None)
        self.network = Word2VecClassifierNetwork()
        if config is not None:
            self.load_config(config)
        else:
            self.path = path
            self.word2vec = load_word2vec_reader(path)
            self.stop_words = stop_words
            self.exploder = Exploder(stop_words)
            self.class_vectors = {}

    @property
    def _axis(self):
        keys = list(self.class_vectors.keys())
        keys.sort()
        return keys

    def _phrase_vector(self, phrase):
        """
        Get phrase word vectors
        :param phrase: phrase
        :type phrase: str
        :return: vectors
        :rtype: list[numpy.ndarray]
        """
        words = self.exploder.explode(phrase)
        vectors = []
        for word in words:
            try:
                vectors.append(self.word2vec[word])
            except KeyError:
                pass
        return vectors

    def _phrase_vec_distance(self, phrase1_vec, phrase2_vec):
        """
        Distance between two phrase words vectors
        :param phrase1_vec: phrase 1 word vectors
        :type phrase1_vec: list[numpy.ndarray]
        :param phrase2_vec: phrase 2 word vectors
        :type phrase2_vec: list[numpy.ndarray]
        :return: distance
        :rtype: float
        """
        def _dist(phrase1_vec, phrase2_vec):
            if len(phrase1_vec) == 0 or len(phrase2_vec) == 0:
                return float("+Inf")
            result = 0.0
            for word1_vec in phrase1_vec:
                diffs = [(word2_vec - word1_vec) for word2_vec in phrase2_vec]
                diff_lengths = [numpy.linalg.norm(diff) for diff in diffs]
                result += min(diff_lengths)
            return result
        return (_dist(phrase1_vec, phrase2_vec) + _dist(phrase2_vec, phrase1_vec)) / 2.0

    def _vec_class_distance(self, phrase_vec, class_name):
        """
        Get distance from phrase word vectors to nearest class item
        :param phrase_vec: phrase words vector
        :type phrase_vec: list[numpy.ndarray]
        :param class_name: class name
        :type class_name: str
        :return: distance
        :rtype: float
        """
        class_vectors = self.class_vectors[class_name]
        diffs = [self._phrase_vec_distance(phrase_vec, class_phrase_vec) \
                 for class_phrase_vec in class_vectors]
        return min(diffs)

    def _vec_classes_distance(self, phrase_vec):
        """
        Get distance from phrase word vectors to classes
        :param phrase_vec: phrase word vectors
        :type phrase_vec: list[numpy.ndarray]
        :return: distances
        :rtype: dict[str, float]
        """
        result = {}
        for item in self._axis:
            result[item] = self._vec_class_distance(phrase_vec, item)
        return result

    def _vec_classes_distance_list(self, phrase_vec):
        """
        Get distance (as list) from phrase word vectors to classes
        :param phrase_vec: phrase word vectors
        :type phrase_vec: list[numpy.ndarray]
        :return: distances
        :rtype: list[float]
        """
        class_distances = self._vec_classes_distance(phrase_vec)
        return [class_distances[cls] for cls in self._axis]

    @property
    def _class_vectors_linear(self):
        """
        :rtype: list[tuple[str, list[numpy.ndarray]]]
        """
        result = []
        for class_name, vectors in self.class_vectors.items():
            for vector in vectors:
                result.append((class_name, vector,))
        return result

    def _fits(self):
        """
        Get fits for neural network
        :return: fits
        :rtype: tuple[list[list[float]], list[list[float]]]
        """
        x = []
        y = []
        for item in self._class_vectors_linear:
            class_name, vector = item
            x.append(self._vec_classes_distance_list(vector))
            y_fit = []
            for other_class in self._axis:
                if other_class == class_name:
                    y_fit.append(1.0)
                else:
                    y_fit.append(0.0)
            y.append(y_fit)
        return x, y

    def train(self, classes):
        """
        Train NLC
        :param classes: classes
        :type classes: Classes
        """
        self.class_vectors = {}
        for class_name, phrases in classes.items():
            phrase_vectors = [self._phrase_vector(phrase) for phrase in phrases]
            self.class_vectors[class_name] = phrase_vectors
        x, y = self._fits()
        self.network.fit(x, y)

    def classify(self, text):
        """
        Classify text
        :param text: text
        :type text: str
        :return: class confidence dict
        :rtype: dict[str, float]
        """
        vec = self._phrase_vector(text)
        distances = self._vec_classes_distance_list(vec)
        confidences = self.network.predict(distances)
        axis = self._axis
        result = {}
        for i in range(0, len(axis)):
            result[axis[i]] = math.pow(confidences[i], 2.5)
        result[''] = 0.1
        length = 0.0
        for value in result.values():
            length += value * value
        length = math.sqrt(length)
        for class_name, confidence in result.items():
            result[class_name] = confidence / length
        del result['']
        return result

    def config(self):
        """
        Get classifier config
        :return: config dict
        :rtype: dict
        """
        class_vectors = {}
        for class_name, values in self.class_vectors.items():
            class_vectors[class_name] = []
            for value in values:
                value_list = [word.tolist() for word in value]
                class_vectors[class_name].append(value_list)
        return {
            'stop_words': self.stop_words,
            'class_vectors': class_vectors,
            'path': self.path,
            'network': self.network.config()
        }

    def load_config(self, config):
        """
        Load configuration
        :param config: config dict
        :type config: dict
        """
        self.path = config['path']
        self.word2vec = load_word2vec_reader(self.path)
        self.stop_words = config['stop_words']
        self.exploder = Exploder(self.stop_words)
        self.network.load_config(config['network'])
        self.class_vectors = {}
        for class_name, vectors in config['class_vectors'].items():
            self.class_vectors[class_name] = []
            for vector in vectors:
                self.class_vectors[class_name].append([numpy.array(word) for word in vector])
