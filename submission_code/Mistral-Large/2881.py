from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator and filter out any empty strings
            split_words = [part for part in word.split(separator) if part]
            result.extend(split_words)
        return result