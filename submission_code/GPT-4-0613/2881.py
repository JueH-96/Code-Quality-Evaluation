class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_word = word.split(separator)
            for w in split_word:
                if w != '':
                    result.append(w)
        return result