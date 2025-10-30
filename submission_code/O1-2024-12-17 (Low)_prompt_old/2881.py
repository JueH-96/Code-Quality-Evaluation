class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the current word by the given separator
            splitted = word.split(separator)
            # Filter out empty strings and extend the result list
            result.extend([part for part in splitted if part])
        return result