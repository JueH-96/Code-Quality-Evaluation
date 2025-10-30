from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Collect all character counts
        total_counts = Counter()
        for word in words:
            total_counts.update(word)
        
        # Compute total_pairs and total_singles
        total_pairs = 0
        total_singles = 0
        for c in total_counts.values():
            total_pairs += c // 2
            total_singles += c % 2
        
        # For each word, compute pairs_needed and singles_needed
        word_info = []
        for word in words:
            l = len(word)
            pairs_needed = l // 2
            singles_needed = 1 if l % 2 else 0
            word_info.append((pairs_needed, singles_needed))
        
        # Sort the words by pairs_needed, then by singles_needed
        word_info.sort()
        
        # Compute prefix sums
        n = len(word_info)
        prefix_pairs = [0] * (n + 1)
        prefix_singles = [0] * (n + 1)
        for i in range(n):
            prefix_pairs[i + 1] = prefix_pairs[i] + word_info[i][0]
            prefix_singles[i + 1] = prefix_singles[i] + word_info[i][1]
        
        max_k = 0
        # Check for all possible k from 0 to n
        for k in range(0, n + 1):
            s = prefix_pairs[k]
            t = prefix_singles[k]
            required_converted = max(0, (t - total_singles + 1) // 2)
            if s + required_converted <= total_pairs:
                max_k = max(max_k, k)
        
        return max_k