from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        counter = Counter()
        for word in words:
            counter += Counter(word)
        return min(counter[ch] for ch in 'abcdefghijklmnopqrstuvwxyz') // 2