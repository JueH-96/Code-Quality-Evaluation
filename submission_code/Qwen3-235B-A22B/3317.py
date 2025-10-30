from collections import defaultdict
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count all characters across all words
        char_count = defaultdict(int)
        lengths = [len(word) for word in words]
        for word in words:
            for c in word:
                char_count[c] += 1
        
        # Calculate initial single characters and total available pairs
        sum_singles = sum(count % 2 for count in char_count.values())
        total_pairs = sum(count // 2 for count in char_count.values())
        
        # Sort lengths to prioritize smaller lengths
        lengths.sort()
        n = len(lengths)
        
        # Precompute prefix sums for pairs and number of odd lengths
        prefix_pairs = [0] * (n + 1)
        prefix_odds = [0] * (n + 1)
        for i in range(n):
            prefix_pairs[i + 1] = prefix_pairs[i] + (lengths[i] // 2)
            prefix_odds[i + 1] = prefix_odds[i] + (lengths[i] % 2)
        
        # Check from maximum possible K down to 0
        for K in range(n, -1, -1):
            if K == 0:
                return 0
            current_pairs = prefix_pairs[K]
            if current_pairs > total_pairs:
                continue
            current_odds = prefix_odds[K]
            available_singles = sum_singles + 2 * (total_pairs - current_pairs)
            if current_odds <= available_singles:
                return K
        
        return 0