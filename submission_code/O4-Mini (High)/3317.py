from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count total occurrences of each character over all words
        total_counts = Counter()
        for w in words:
            total_counts.update(w)
        
        # Total number of pairs available from the pooled characters
        total_pairs = sum(cnt // 2 for cnt in total_counts.values())
        
        # For each word, floor(len/2) is how many pairs it needs to become a palindrome
        pair_requirements = [len(w) // 2 for w in words]
        pair_requirements.sort()
        
        # Greedily take as many words as possible starting from the smallest pair needs
        used_pairs = 0
        result = 0
        for need in pair_requirements:
            if used_pairs + need <= total_pairs:
                used_pairs += need
                result += 1
            else:
                break
        
        return result