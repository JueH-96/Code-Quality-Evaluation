from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        # Return the list of indices where character x appears in the word
        return [i for i, w in enumerate(words) if x in w]