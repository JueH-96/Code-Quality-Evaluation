class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for w in words:
            parts = w.split(separator)
            for part in parts:
                if part:
                    result.append(part)
        return result