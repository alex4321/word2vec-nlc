class Classes(dict):
    """
    NLC classes examples
    """

    def __setitem__(self, key, value):
        """
        Set class examples
        :param key: class name
        :type key: str
        :param value: class items
        :type value: list[str]
        """
        assert isinstance(key, str)
        assert isinstance(value, list)
        for item in value:
            assert isinstance(item, str)
        super(Classes, self).__setitem__(key, value)

    def __getitem__(self, item):
        """
        Get class examples
        :param item: class name
        :type item: str
        :return: examples
        :rtype: list[str]
        """
        assert isinstance(item, str)
        return super(Classes, self).__getitem__(item)
