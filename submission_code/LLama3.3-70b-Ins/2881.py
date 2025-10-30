from typing import List

class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        result = []
        for word in words:
            # Split the word by the separator and filter out empty strings
            split_word = [w for w in word.split(separator) if w != '']
            # Add the split words to the result list
            result.extend(split_word)
        return result