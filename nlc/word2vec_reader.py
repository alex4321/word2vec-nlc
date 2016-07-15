import struct
import numpy


class Word2VecReader(dict):
    """
    Word2Vec model reader (not training - only reading!)
    """
    def __init__(self, path):
        """
        Initialize word2vec reader
        :param path: path to model binary file
        :type path: str
        """
        self._load(path)
        super(Word2VecReader, self).__init__({})

    def _read_until(self, f, splitters):
        """
        Read until given bytes
        :type f: file
        :type splitters: bytearray
        :rtype: bytearray
        """
        result = []
        while True:
            readed = f.read(1)[0]
            if readed in splitters:
                break
            result.append(readed)
        return bytearray(result)

    def _load(self, path):
        """
        Load model
        :param path: path to model binary file
        :type path: str
        """
        with open(path, 'rb') as f:
            header_parts = str(self._read_until(f, [ord('\n')]).decode('utf-8')).split(' ')
            self._length, self._size = [int(item) for item in header_parts]
            for i in range(0, self._length):
                word = self._read_until(f, [ord(' ')]).decode('utf-8')
                values = []
                for j in range(0, self._size):
                    values.append(struct.unpack('f', f.read(4))[0])
                self[word] = numpy.array(values)

    def __setitem__(self, key, value):
        """
        Set word vector
        :param key: word
        :type key: str
        :param value: vector
        :type value: numpy.ndarray
        """
        assert isinstance(key, str)
        assert isinstance(value, numpy.ndarray)
        super(Word2VecReader, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get word vector
        :param item: word
        :type item: str
        :return: vector
        :rtype: numpy.ndarray
        """
        assert isinstance(item, str)
        return super(Word2VecReader, self).__getitem__(item)
