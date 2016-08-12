word2vec-nlc
------------
This is Keras and word2vec-based NLC. It use "interface" [from my other library](https://github.com/alex4321/nlc).
It just make matrix from word2vec vectors of not-stop words and train LSTM network on it.

Installation
------------
Firstly you need to install my library [from github](https://github.com/alex4321/nlc). Also, you must have installed [keras](https://keras.io/) and backend for it (I used [theano](http://deeplearning.net/software/theano/)). 

```
pip install keras theano
```

Using
-----
See example in word2vec_classifier_test.py and nlc/classifier_test.py.
Also, maybe in your case you'll need to [use GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)