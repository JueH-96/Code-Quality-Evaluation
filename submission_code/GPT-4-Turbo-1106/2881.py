class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator and extend the result list with non-empty strings
            result.extend([s for s in word.split(separator) if s])
        return result