from collections import Counter
from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        freq = Counter(words)
        count = 0
        for word in words:
            if freq[word] == 0:
                continue
            reversed_word = word[::-1]
            if reversed_word == word:
                continue  # Skip palindromes as they can't form pairs with distinct words
            if reversed_word in freq and freq[reversed_word] > 0:
                count += 1
                freq[word] -= 1
                freq[reversed_word] -= 1
        return count