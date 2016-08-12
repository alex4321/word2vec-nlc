import os
from unittest import TestCase
from .word2vec_loader import load as load_word2vec, get_path as get_word2vec_path


class LoaderTest(TestCase):
    def testLoading(self):
        path = os.path.join(os.path.dirname(__file__), "test", "glove.6B.50d.bin")
        model1 = load_word2vec(path)
        model2 = load_word2vec(path)
        self.assertEqual(hash(model1), hash(model2))

    def testPath(self):
        path = os.path.join(os.path.dirname(__file__), "test", "glove.6B.50d.bin")
        model = load_word2vec(path)
        self.assertEqual(path, get_word2vec_path(model))
