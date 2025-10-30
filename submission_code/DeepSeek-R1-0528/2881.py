class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        return [token for word in words for token in word.split(separator) if token]