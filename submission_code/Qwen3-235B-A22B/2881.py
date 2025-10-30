from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            parts = word.split(separator)
            filtered = [part for part in parts if part]
            result.extend(filtered)
        return result