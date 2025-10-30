from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator
            parts = word.split(separator)
            # Filter out empty strings and extend the result list
            result.extend(part for part in parts if part)
        return result