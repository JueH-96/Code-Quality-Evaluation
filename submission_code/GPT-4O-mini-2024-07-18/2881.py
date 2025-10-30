from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            split_words = word.split(separator)
            result.extend([w for w in split_words if w])  # Exclude empty strings
        return result