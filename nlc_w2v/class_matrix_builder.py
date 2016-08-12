from itertools import permutations
import numpy
from gensim.models import Word2Vec
from .exploder import Exploder


class ClassMatrixBuilder:
    """
    Build matrix for NLC classification.
    E.g. at input we have
    { 'class1': ['sentence 1', 'sentence 2'],
      'class2': ['sentence 3', 'sentence 4'] }
    And it'll build
    {
      'class1': [
        [SentenceWordVector, 1Vector],
        [SentenceWordVector, 2Vector],
      ],
      'class2': [
        [SentenceWordVector, 3Vector],
        [SentenceWordVector, 4Vector],
      ],
    }
    """

    def __init__(self, word2vec, exploder):
        """
        Initialize matrix builder
        :param word2vec: word2vec model
        :type word2vec: Word2Vec
        :param exploder: exploder
        :type exploder: Exploder
        """
        self.word2vec = word2vec
        self.exploder = exploder
        self.sentence_matrix_length = 0

    def _sentence_matrix(self, words):
        """
        Get sentence matrix
        :param words: words
        :type words: list[str]
        :return: matrix of [word, wordVectorComponent]
        :rtype: numpy.ndarray
        """
        result = []
        for word in words:
            try:
                result.append(self.word2vec[word])
            except KeyError:
                result.append(numpy.zeros(self.word2vec.vector_size))
        return numpy.array(result)

    def _class_examples_matrix_normalize_size(self, matrixes, length, verbose):
        """
        Chang length of sentence matrixes to same value (by adding leading zero rows)
        :param matrixes: sentence matrixes
        :type matrixes: list[numpy.ndarray]
        :param length: target length
        :type length: int
        :param verbose: verbose?
        :type verbose: bool
        :return: resized matrix
        :rtype: numpy.ndarray
        """
        sized_result = []
        for i, matrix in enumerate(matrixes):
            add_rows = length - len(matrix)
            if verbose:
                print("Resize base matrix for {0}/{1} example".format(i + 1, len(matrixes)))
            if add_rows > 0:
                empty_rows = numpy.zeros((add_rows, self.word2vec.vector_size,))
                matrix_linear = numpy.append(matrix, empty_rows)
                matrix = matrix_linear.reshape((length, self.word2vec.vector_size,))
            sized_result.append(matrix)
        return numpy.array(sized_result)

    def _class_examples_matrix(self, examples, verbose):
        """
        Class examples matrix
        :param examples: examples
        :type examples: list[str]
        :param verbose: verbose
        :type verbose:  bool
        :return: example matrix (indexes - exampleNumber, wordNumber, wordVectorComponentNumber)
        :rtype: numpy.ndarray
        """
        result = []
        max_length = 0
        processed = []
        for i, example in enumerate(examples):
            if verbose:
                print("Building base matrix for {0}/{1} example".format(i + 1, len(examples)))
            words = set(self.exploder.explode(example))
            try:
                processed.index(words)
            except ValueError:
                processed.append(words)
                word_permutations = list(permutations(words))
                for word_permutation in word_permutations:
                    word_permutation_list = list(word_permutation)
                    matrix = self._sentence_matrix(word_permutation_list)
                    result.append(matrix)
                    if len(matrix) > max_length:
                        max_length = len(matrix)
        return self._class_examples_matrix_normalize_size(
            result, max_length, verbose
        )

    def class_sentence_matrixs(self, classes, verbose):
        """
        Build classes matrixes
        :param classes: classes
        :type classes: dict[str, list[str]]
        :param verbose: verbose
        :type verbose: bool
        :return: matrixes
        :rtype: dict[str, numpy.ndarray]
        """
        result = {}
        lengths = []
        for class_name, examples in classes.items():
            if verbose:
                print("Building base matrixes for {0}".format(class_name))
            matrix = self._class_examples_matrix(examples, verbose)
            shape = list(matrix.shape)[1]
            lengths.append(shape)
            result[class_name] = matrix
        length = max(lengths)
        self.sentence_matrix_length = length
        sized_result = {}
        for class_name, examples in result.items():
            if verbose:
                print("Building resized matrixes for {0}".format(class_name))
            examples_resized = []
            for example in examples:
                add_rows = length - len(example)
                if add_rows == 0:
                    examples_resized.append(example)
                else:
                    empty_rows = numpy.zeros((add_rows, self.word2vec.vector_size,))
                    example_linear = numpy.append(example, empty_rows)
                    examples_resized.append(
                        example_linear.reshape((length, self.word2vec.vector_size,))
                    )
            sized_result[class_name] = numpy.array(examples_resized)
        return sized_result

    def phrase_matrix(self, text):
        """
        Get phrase matrix
        :param text: text
        :type text: str
        :return: matrix
        :type: list[list[float]]
        """
        words = self.exploder.explode(text)
        return self._sentence_matrix(words).tolist()
