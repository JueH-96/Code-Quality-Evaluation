class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            parts = word.split(separator)
            for part in parts:
                if part:  # This checks if part is non-empty
                    result.append(part)
        return result