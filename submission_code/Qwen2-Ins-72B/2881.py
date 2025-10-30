class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_words = word.split(separator)
            for w in split_words:
                if w: # Check if the string is not empty
                    result.append(w)
        return result