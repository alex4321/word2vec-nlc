from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy


class Word2VecClassifierNetwork:
    """
    Classifier neural network (converts class distances to class confidences)
    """
    def __init__(self):
        self.class_count = 0
        self.model = Sequential()

    def _resize(self, class_count):
        """
        Build empty model with given dimension
        :param class_count: class count
        :type class_count: int
        """
        self.model = Sequential()
        self.class_count = class_count
        self.model.add(Dense(400 * self.class_count, input_dim=self.class_count))
        self.model.add(Dense(class_count, activation='softmax'))
        optimizer = SGD(lr=0.002, momentum=0.0, decay=0.0, nesterov=True)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

    def fit(self, x, y):
        """
        Train model with given data.
        Where x[i] - class distances, y[i] - class confidences
        :param x: distances
        :type x: list[list[float]]
        :param y: confidences
        :type y: list[list[float]]
        """
        self._resize(len(x[0]))
        self.model.fit(numpy.array(x), numpy.array(y), nb_epoch=40, verbose=0)

    def predict(self, x):
        """
        Get confidences for given distances
        :param x: class distances
        :type x: list[float]
        :return: confidences
        :rtype: list[float]
        """
        return self.model.predict(numpy.array([x]))[0]

    def config(self):
        """
        Get configuration
        :return: config
        :rtype: dict
        """
        return {
            'model': self.model.get_config(),
            'weights': self.model.get_weights()
        }

    def load_config(self, config):
        """
        Initialize from config
        :param config: config
        :type config: dict
        """
        self.model = Sequential.from_config(config['model'])
        self.model.set_weights(config['weights'])
