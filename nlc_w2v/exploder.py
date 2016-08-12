class Exploder:
    """
    Words extractor
    """

    def __init__(self, stop_words, delimeters="`~@\"#№$;:%^&?*()-_=+<>,.\\|/[]{} \r\n\t"):
        """
        Initialize exploder
        :param stop_words: stopwords
        :type stop_words: list[str]
        :param delimeters: delimeter chars
        :type delimeters: str
        """
        self.stop_words = stop_words
        self.delimeters = delimeters

    def explode(self, text):
        """
        Explode text
        :param text: text
        :type text: str
        :return: words
        :rtype: list[str]
        """
        items = [""]
        for char in text.lower():
            if char in self.delimeters:
                items.append("")
            else:
                items[-1] += char
        items = [item for item in items if item != ""]
        return [item for item in items if item not in self.stop_words]
