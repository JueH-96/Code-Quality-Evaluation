from collections import defaultdict
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Calculate the total frequency of each character across all words
        freq = defaultdict(int)
        for word in words:
            for c in word:
                freq[c] += 1
        
        # Compute S: the number of characters with odd counts
        S = sum(1 for count in freq.values() if count % 2 != 0)
        
        # Count the number of even-length (E) and odd-length (O) words
        E = 0
        O = 0
        for word in words:
            if len(word) % 2 == 0:
                E += 1
            else:
                O += 1
        
        max_palindromes = 0
        max_x = min(O, S)
        
        # Iterate over possible x (number of odd-length words turned into palindromes)
        for x in range(0, max_x + 1):
            assigned_even = min(S - x, E)
            even_palindromes = E - assigned_even
            current_total = x + even_palindromes
            if current_total > max_palindromes:
                max_palindromes = current_total
        
        return max_palindromes