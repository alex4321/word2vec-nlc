from unittest import TestCase
from .word2vec import Word2Vec

class Word2VecTest(TestCase):
    def setUp(self):
        self.model = Word2Vec("test/GoogleNews-vectors-negative300.bin")

    #def test_model_equality(self):
    #    self.assertIs(self.model.model, Word2Vec("test/GoogleNews-vectors-negative300.bin").model)

    def test_nearest(self):
        demo = self.model.similar(positive=['I', 'have', 'computer'])
        input = self.model.similar(positive=['I', 'have', 'laptop'])
        wrong = self.model.similar(positive=['It', 'dog'])
        demo_distance = demo - input
        wrong_distance = wrong - input
        print(demo_distance, demo_distance.length())
        print(wrong_distance, wrong_distance.length())
