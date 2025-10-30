from collections import Counter
from itertools import chain

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character in all words
        char_freq = Counter(chain(*words))
        
        # Count how many characters have an even frequency and how many have an odd frequency
        even_chars = sum(1 for char, freq in char_freq.items() if freq % 2 == 0)
        odd_chars = sum(1 for char, freq in char_freq.items() if freq % 2 != 0)
        
        # Calculate the maximum number of palindromes
        max_palindromes = 0
        if even_chars > 0:
            max_palindromes += even_chars // 2
        if odd_chars > 0:
            max_palindromes += (odd_chars + 1) // 2
        
        return min(max_palindromes, len(words))