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
        :type value: list[Vector]
        """
        assert isinstance(key, str)
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, Vector)
        super(ClassVectors, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get class items vectors
        :param item: class name
        :type item: str
        :return: items
        :rtype: list[Vector]
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
        self.class_vectors = ClassVectors({})
        for name in classes.keys():
            phrases = classes[name]
            phrase_vectors = [word2vec.similar(self.extractor.extract(phrase)) for phrase in phrases]
            self.class_vectors[name] = phrase_vectors

    def classify(self, text):
        """
        Classify user input
        :param text: source text
        :type text: str
        :return: classification result
        :rtype: list[ClassifierResultItem]
        """
        vector = self.word2vec.similar(self.extractor.extract(text))
        class_diff_lengths = {}
        for class_name in self.class_vectors:
            class_vectors = self.class_vectors[class_name]
            for class_vector in class_vectors:
                diff_vector = vector - class_vector
                if class_name not in class_diff_lengths.keys() or diff_vector.length() < class_diff_lengths[class_name]:
                    class_diff_lengths[class_name] = diff_vector.length()
        result = []
        print(class_diff_lengths)
        for class_name in class_diff_lengths.keys():
            diff_length = class_diff_lengths[class_name]
            result.append(ClassifierResultItem(class_name, diff_length, 0))
        result.sort()
        return result
