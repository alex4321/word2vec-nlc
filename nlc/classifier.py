from .word2vec import Word2Vec, Vector
from .extractor import Extractor


class Classes(dict):
    def __setitem__(self, key, value):
        """
        Set class items
        :param key: class name
        :type key: str
        :param value: class examples
        :type value: list[str]
        """
        assert isinstance(key, str)
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, str)
        super(Classes, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get class items
        :param item: class name
        :type item: str
        :return: items
        :rtype: list[str]
        """
        assert isinstance(item, str)
        return super(Classes, self).__getitem__(item)


class ClassVectors(dict):
    def __setitem__(self, key, value):
        """
        Set class vectors
        :param key: class name
        :type key: str
        :param value: class example vectors
        :type value: Vector
        """
        assert isinstance(key, str)
        assert isinstance(value, Vector)
        super(ClassVectors, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get class items vectors
        :param item: class name
        :type item: str
        :return: items
        :rtype: Vector
        """
        assert isinstance(item, str)
        return super(ClassVectors, self).__getitem__(item)


class ClassifierResultItem:
    def __init__(self, class_name, distance, confidence):
        """
        Classifier result item
        :param class_name: class name
        :type class_name: str
        :param distance: class distance
        :type distance: float|int
        :param confidence: confidence
        :type confidence: float|int
        """
        self.class_name = class_name
        self.distance = distance
        self.confidence = confidence

    def __cmp__(self, other):
        assert isinstance(other, ClassifierResultItem)
        if self.distance < other.distance:
            return -1
        elif self.distance > other.distance:
            return 1
        else:
            return 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0


class Classifier:
    def __init__(self, word2vec, language, classes):
        """
        Initialize NLC
        :param word2vec: Word2Vec model
        :type word2vec: Word2Vec
        :param language: language code
        :type language: str
        :param classes: Classes
        :type classes: Classes
        """
        self.word2vec = word2vec
        self.extractor = Extractor(language)
        self._build_clusters(classes)
        self._calculate_center()

    def _build_clusters(self, classes):
        """
        :param classes: Classes
        :type classes: Classes
        """
        self.clusters = ClassVectors({})
        for name in classes.keys():
            phrases = classes[name]
            phrase_vectors = [self.word2vec.similar(self.extractor.extract(phrase)) for phrase in phrases]
            middle_vector = Vector({})
            for vector in phrase_vectors:
                middle_vector += vector
            middle_vector /= float(len(phrase_vectors))
            self.clusters[name] = middle_vector

    def _calculate_center(self):
        self.center = Vector({})
        for class_name in self.clusters.keys():
            self.center += self.clusters[class_name]
        self.center /= len(self.clusters)

    def _confidence(self, vector, class_name):
        """
        :type vector: Vector
        :type class_name: str
        :rtype: float
        """
        vector_to_class = vector - self.clusters[class_name]
        class_to_center = self.clusters[class_name] - self.center
        no_confidence = vector_to_class.length() / class_to_center.length()
        if no_confidence > 1:
            no_confidence = 1
        return 1 - no_confidence

    def _class_diff(self, vector, class_name):
        """
        :type vector: Vector
        :type class_name: str
        :rtype: Vector
        """
        diff_vector = self.clusters[class_name] - vector
        return diff_vector


    def classify(self, text):
        """
        Classify user input
        :param text: source text
        :type text: str
        :return: classification result
        :rtype: list[ClassifierResultItem]
        """
        vector = self.word2vec.similar(self.extractor.extract(text))
        result = []
        for class_name in self.clusters.keys():
            result.append(ClassifierResultItem(
                class_name,
                self._class_diff(vector, class_name).length(),
                self._confidence(vector, class_name)))
        result.sort()
        return result
