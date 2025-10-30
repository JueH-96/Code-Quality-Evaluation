from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the given separator
            parts = word.split(separator)
            # Filter out any empty strings
            filtered_parts = [part for part in parts if part]
            # Extend the result list with the filtered parts
            result.extend(filtered_parts)
        return result