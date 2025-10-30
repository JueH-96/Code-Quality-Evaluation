from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator and filter out empty strings
            parts = [s for s in word.split(separator) if s]
            result.extend(parts)
        return result