class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            parts = word.split(separator)
            # Include only non-empty strings
            for part in parts:
                if part:
                    result.append(part)
        return result