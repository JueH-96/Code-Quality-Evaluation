from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        char_count = Counter()
        for word in words:
            char_count.update(word)
        
        even_pairs = sum(count // 2 for count in char_count.values())
        word_lengths = sorted(len(word) for word in words)
        
        palindromes = 0
        for length in word_lengths:
            if even_pairs >= length // 2:
                even_pairs -= length // 2
                palindromes += 1
            else:
                break
        
        return palindromes