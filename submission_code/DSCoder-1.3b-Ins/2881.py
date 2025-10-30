class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_word = word.split(separator)
            split_word = [i for i in split_word if i]
            result.extend(split_word)
        return result