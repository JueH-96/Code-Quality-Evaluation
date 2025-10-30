from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the given separator
            parts = word.split(separator)
            # Append only non-empty parts to the result
            for part in parts:
                if part:
                    result.append(part)
        return result