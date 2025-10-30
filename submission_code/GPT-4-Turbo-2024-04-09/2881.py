class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator and filter out empty strings
            parts = [part for part in word.split(separator) if part]
            result.extend(parts)
        return result