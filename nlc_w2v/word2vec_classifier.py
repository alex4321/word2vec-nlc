from collections import OrderedDict
import math
from itertools import permutations
import numpy
from gensim.models import Word2Vec
from nlc import BaseClassifier
from .exploder import Exploder
from .class_solver import ClassSolver
from .word2vec_loader import load as load_word2vec, get_path as get_word2vec_path
from .class_matrix_builder import ClassMatrixBuilder


class Word2VecClassifier(BaseClassifier):
    """
    Classifier with word2vec-based distances
    Concepts:
    sentence - part of text
    sentence matrix - matrix with next view:
      [ [word_1_1, ... word_1_n],
        ...
        [word_m_1, ... word_2_n]]
      where [word_k_1, ... word_k_n] - word2vec vector
    class - group of text examples
    class distance - distance to nearest sentence matrix of given class
    matrix to matrix distance - sqrt(sum(matrix1 - matrix2 ^ 2))
    classes distance - distance to all stored classes
    class confidence - value in [0.0:1.0] (0.0 - sentence is n't part of given class, 1.0 - is part)
    """

    def __init__(self, word2vec=None, exploder=None,
                 word2vec_path=None, exploder_stop_words=None,
                 class_sentence_matrixed=None,
                 confidence_converter_config=None,
                 sentence_matrix_length=0):
        """
        :param word2vec: word2vec model
        :type word2vec: Word2Vec|NoneType
        :param exploder: exploder
        :type exploder: Exploder|NoneType
        """
        if exploder_stop_words is not None:
            exploder = Exploder(exploder_stop_words)
        elif exploder is None:
            raise AttributeError("Need exploder or exploder_stop_words")
        if word2vec is not None:
            self.matrix_builder = ClassMatrixBuilder(word2vec, exploder)
            self.word2vec_path = get_word2vec_path(word2vec)
        elif word2vec_path is not None:
            self.matrix_builder = ClassMatrixBuilder(load_word2vec(word2vec_path), exploder)
            self.word2vec_path = word2vec_path
        else:
            raise AttributeError("Need word2vec or word2vec_path")
        self.class_sentence_matrixed = {}
        if class_sentence_matrixed is not None:
            for key, matrix in class_sentence_matrixed.items():
                self.class_sentence_matrixed[key] = numpy.array(matrix)
        if confidence_converter_config is None:
            self.confidence_converter = ClassSolver(0)
        else:
            self.confidence_converter = ClassSolver(**confidence_converter_config)
        self.matrix_builder.sentence_matrix_length = sentence_matrix_length

    @property
    def _axis(self):
        """
        Axis class names
        :return: axis
        :rtype: list[str]
        """
        lst = list(self.class_sentence_matrixed.keys())
        lst.sort()
        return lst

    def _class_matrix_distance(self, class_name, matrix, trim_zeros=False):
        """
        Get distance from matrix to class matrix
        :param class_name: class name
        :type class_name: str
        :param matrix: matrix
        :type matrix: numpy.ndarray
        :param trim_zeros: trim zero distances?
        :type trim_zeros: bool
        :return: distance
        :rtype: float
        """
        # Get list of difference between class record and given
        diffs = self.class_sentence_matrixed[class_name] - matrix
        diffs_sqr = diffs * diffs
        diff_sums = numpy.sum(diffs_sqr, (1, 2,))  # Transform difference to linear
        if trim_zeros:
            diff_sums = numpy.trim_zeros(diff_sums)
        return math.sqrt(diff_sums.min())  # Get distance

    def _matrix_distances(self, matrix):
        """
        Get distance from matrix to classes
        :param matrix: matrix
        :type matrix: numpy.ndarray
        :return: distances
        :rtype: dict[str, float]
        """
        result = {}
        for class_name in self.class_sentence_matrixed:
            result[class_name] = self._class_matrix_distance(class_name, matrix, True)
        return result

    def _fits(self, verbose):
        """
        Get distance to confidence converter train fits
        :param verbose: verbose process?
        :type verbose: bool
        :return: examples
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        distances = []
        confidences = []
        for class_name, class_matrix in self.class_sentence_matrixed.items():
            if verbose:
                print("Building fits for {0}".format(class_name))
            for i, item in enumerate(class_matrix):
                if verbose:
                    print("Building fit {0} of {1}".format(i+1, len(class_matrix)))
                class_distances = self._matrix_distances(item)
                x_fit = []
                y_fit = []
                for axis in self._axis:
                    x_fit.append(class_distances[axis])
                    if axis == class_name:
                        y_fit.append(1.0)
                    else:
                        y_fit.append(0.0)
                distances.append(x_fit)
                confidences.append(y_fit)
        return numpy.array(distances), numpy.array(confidences)

    def train(self, classes, verbose=False):
        """
        Train NLC
        :param classes: classes
        :type classes: dict[str, list[str]]
        :param verbose: verbose
        :type verbose: bool
        """
        if verbose:
            print("Building class matrixes")
        self.class_sentence_matrixed = self.matrix_builder.class_sentence_matrixs(classes, verbose)
        distances, confidences = self._fits(verbose)
        self.confidence_converter.resize(len(classes))
        self.confidence_converter.train(distances, confidences, verbose)

    def classify(self, text):
        """
        Classify text
        :param text: text
        :type text: str
        :return: classes
        :rtype: OrderedDict[str, float]
        """
        def class_confidence_pair_comparer(pair):
            _, conf = pair
            return conf
        matrix = self.matrix_builder.phrase_matrix(text)
        class_distances = []
        for permutation in [list(item) for item in permutations(matrix)]:
            permutation_part = numpy.array(permutation[:self.matrix_builder.sentence_matrix_length])
            add_rows = self.matrix_builder.sentence_matrix_length - len(permutation_part)
            right_shape = (self.matrix_builder.sentence_matrix_length, self.matrix_builder.word2vec.vector_size,)
            if add_rows > 0:
                empty_rows = numpy.zeros((add_rows, self.matrix_builder.word2vec.vector_size,))
                permutation_part = numpy.append(permutation_part, empty_rows)\
                    .reshape(right_shape)
            matrix_distances = self._matrix_distances(permutation_part)
            class_distances.append([matrix_distances[axis] for axis in self._axis])
        class_distances = numpy.array(class_distances)
        distances = []
        for i in range(0, len(self._axis)):
            column = class_distances[:, i]
            distances.append(column.min())
        confidences = self.confidence_converter.calculate(numpy.array(distances))
        class_confidence_pairs = []
        for i, class_name in enumerate(self._axis):
            class_confidence_pairs.append((class_name, confidences[i],))
        class_confidence_pairs.sort(key=class_confidence_pair_comparer, reverse=True)
        result = OrderedDict()
        for item in class_confidence_pairs:
            class_name, confidence = item
            result[class_name] = confidence
        return result

    def _get_config(self):
        """
        Get config dict
        :return: config
        :rtype: dict
        """
        class_sentence_matrixed = {}
        for class_name, matrix in self.class_sentence_matrixed.items():
            class_sentence_matrixed[class_name] = matrix.tolist()
        return {
            'word2vec_path': self.word2vec_path,
            'exploder_stop_words': self.matrix_builder.exploder.stop_words,
            'class_sentence_matrixed': class_sentence_matrixed,
            'confidence_converter_config': self.confidence_converter.config,
            'sentence_matrix_length': self.matrix_builder.sentence_matrix_length
        }

BaseClassifier.register('word2vec', Word2VecClassifier)
