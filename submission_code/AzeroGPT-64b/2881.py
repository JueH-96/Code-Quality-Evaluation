class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            result += [s for s in word.split(separator) if s]  # Use list comprehension to exclude empty strings
        return result