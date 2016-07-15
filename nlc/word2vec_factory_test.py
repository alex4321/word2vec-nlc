import unittest
import os
from .word2vec_factory import load_word2vec_reader


class Word2VecFactoryTest(unittest.TestCase):
    """
    Word2Vec model factory test
    """

    def test_eq(self):
        """
        Test instanse equalence
        """
        path = os.path.join(os.path.dirname(__file__), "test", "glove.6B.50d.bin")
        inst1 = load_word2vec_reader(path)
        inst2 = load_word2vec_reader(path)
        self.assertIs(inst1, inst2)
