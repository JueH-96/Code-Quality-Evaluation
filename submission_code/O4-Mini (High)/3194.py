from typing import List

class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        # Collect indices i where the character x appears in words[i]
        return [i for i, w in enumerate(words) if x in w]