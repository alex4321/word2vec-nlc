import math
import keras
import numpy


class ClassSolver:
    """
    Class distance to confidence converted
    Class confidences calculated as :
    preconfidence(class_weights) = class_weights[0] + sum(class_distance[i] * class_weights[i+1])
    preconfidence(FAKE) = 0.1
    confidence = preconfidence.normal() ^ 4
    """
    _FAKE_LEN = 0.1

    def __init__(self, class_count, weights=None):
        """
        Initialize new empty solver
        :param class_count: class count
        :type class_count: int
        """
        self.class_count = 0
        self.model = keras.models.Sequential()
        self.resize(class_count)
        if weights is not None:
            self.model.weights = weights

    def resize(self, class_count):
        """
        Resize solver. Will descruct old weights
        :param class_count: class count
        :type class_count: int
        """
        self.class_count = class_count
        self.model.layers.clear()
        if class_count > 0:
            self.model.add(keras.layers.Dense(10 * self.class_count, input_shape=(self.class_count,), activation='softmax'))
            self.model.add(keras.layers.Dense(self.class_count, activation='softmax'))
            self.model.compile('sgd', 'mse', metrics=['accuracy'])

    @property
    def weights(self):
        return self.model.get_weights()

    @weights.setter
    def _set_weights(self, weights):
        self.model.set_weights(weights)

    def calculate(self, distances):
        """
        Get calculated confidences.
        Get vector of every class pre-confidences (see _class_precalc)
        and :
        - normalize this vector
        - square it
        _RESULT_POW times
        :param distances: class distances
        :type distances: numpy.ndarray
        :return: confidences
        :rtype: numpy.ndarray
        """
        predicted = self.model.predict(numpy.array([distances]))[0]
        vec = predicted * predicted
        size = numpy.linalg.norm(vec) + math.pow(ClassSolver._FAKE_LEN, 2.0)
        return vec / size

    def _calculate_class(self, distances, class_index):
        """
        Get one class confidence (see calculate)
        :param distances: class distances
        :type distances: numpy.ndarray
        :param class_index: class index
        :type class_index: int
        :return: confidences
        :rtype: float
        """
        return self.calculate(distances)[class_index]

    def _class_error(self, class_index, distances, confidences):
        """
        Get class training error
        :param class_index: class index
        :type class_index: int
        :param distances: distances
        :type distances: numpy.ndarray
        :param confidences: confidences
        :type confidences: numpy.ndarray
        :return: error
        :rtype: float
        """
        right = confidences[:, class_index]
        real = []
        for row in distances:
            real.append(self._calculate_class(row, class_index))
        diff = right - numpy.array(real)
        diff_sqr = diff * diff
        return math.sqrt((diff_sqr / len(diff_sqr)).sum())

    def train(self, distances, confidences, verbose=False):
        """
        Train solver
        :param distances: distances
        :type distances: numpy.ndarray
        :param confidences: confidences
        :type confidences: numpy.ndarray
        :param verbose: show train information?
        :type verbose: bool
        """
        self.model.fit(distances, confidences, verbose={True: 1, False: 0}[verbose])

    @property
    def config(self):
        """
        Get config
        :return: config
        :rtype: dict
        """
        return {
            'class_count': self.class_count,
            'weights': [
                item.tolist() for item in self.weights
            ]
        }
