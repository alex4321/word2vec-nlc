from .classes import Classes


class Classifier:
    """
    Base abstract NLC class
    """

    def train(self, classes):
        """
        Train NLC
        :param classes: classes
        :type classes: Classes
        """
        raise NotImplementedError()

    def classify(self, text):
        """
        Classify text
        :param text: text
        :type text: str
        :return: class confidence dict
        :rtype: dict[str, float]
        """
        raise NotImplementedError()

    def config(self):
        """
        Get classifier config
        :return: config dict
        :rtype: dict
        """
        raise NotImplementedError()

    def load_config(self, config):
        """
        Load configuration
        :param config: config dict
        :type config: dict
        """
        raise NotImplementedError()
