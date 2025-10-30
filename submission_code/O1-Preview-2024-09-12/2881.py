class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            substrings = word.split(separator)
            for s in substrings:
                if s != '':
                    result.append(s)
        return result