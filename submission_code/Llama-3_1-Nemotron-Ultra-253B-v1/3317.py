from typing import List
from collections import defaultdict

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Calculate the total count of each character
        char_count = defaultdict(int)
        for word in words:
            for c in word:
                char_count[c] += 1
        
        # Calculate total pairs and original odd counts
        total_pairs = sum(v // 2 for v in char_count.values())
        original_odds = sum(v % 2 for v in char_count.values())
        
        # Prepare list of (p, s) for each word
        word_requirements = []
        for word in words:
            length = len(word)
            p = length // 2
            s = 1 if length % 2 else 0
            word_requirements.append((p, s))
        
        # Sort by p ascending, then s ascending
        word_requirements.sort(key=lambda x: (x[0], x[1]))
        
        current_p = 0
        current_s = 0
        max_palindromes = 0
        
        for p, s in word_requirements:
            if current_p + p <= total_pairs and current_s + s <= original_odds:
                current_p += p
                current_s += s
                max_palindromes += 1
        
        return max_palindromes