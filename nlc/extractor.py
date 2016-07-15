from .blacklist import BLACKLIST

class Extractor:
    def __init__(self, language, splitters = "`~!@\"#№$;:'&?*()-_=+<>.,/\\|{}[] \r\n\t"):
        """
        Extract meaning words
        :param language: language code
        :type language: str
        :param splitters: split characters
        :type splitters: str
        """
        self.language = language
        self.splitters = splitters

    @property
    def _blacklist(self):
        return [item.lower() for item in BLACKLIST[self.language]]

    def _filter(self, items):
        """
        :type items: list[str]
        :rtype: list[str]
        """
        result = []
        for item in items:
            if item != '' and item not in self._blacklist:
                result.append(item.lower())
        return result

    def extract(self, text):
        """
        Extract words from text
        :param text: source text
        :type text: str
        :return: list of words
        :rtype: list[str]
        """
        assert isinstance(text, str)
        result = ['']
        for char in text:
            if char not in self.splitters:
                result[-1] += char
            else:
                result.append('')
        return self._filter(result)
