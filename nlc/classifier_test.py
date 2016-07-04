from .classifier import Classes, Classifier
from .word2vec import Word2Vec
from unittest import TestCase


class ClassifierTest(TestCase):
    def setUp(self):
        model = Word2Vec("test/GoogleNews-vectors-negative300.bin")
        classes = Classes({
            'computer': ['I have a computer', 'I have a laptop'],
            'dog': ['I have a dog']
        })
        self.classifier = Classifier(model, 'en', classes)

    def test_classification(self):
        classes = self.classifier.classify('Do you have a cat?')
        self.assertTrue(len(classes) > 0)
        self.assertEqual(classes[0].class_name, 'computer')
