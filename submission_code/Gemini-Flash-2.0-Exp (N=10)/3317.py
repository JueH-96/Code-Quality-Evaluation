from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        char_counts = Counter()
        for word in words:
            char_counts.update(word)
        
        odd_count = 0
        for count in char_counts.values():
            odd_count += count % 2
        
        
        word_lengths = sorted([len(word) for word in words])
        
        palindromes = 0
        
        for length in word_lengths:
            if length % 2 == 0:
                if odd_count >= 0:
                    palindromes += 1
            else:
                if odd_count > 0:
                    palindromes += 1
                    odd_count -= 1
        
        return palindromes