class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            temp = word.split(separator)
            for t in temp:
                if t:  # if t is not an empty string
                    result.append(t)
        return result