class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_words = word.split(separator)
            for subword in split_words:
                if subword:
                    result.append(subword)
        return result