from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        char_counts = Counter()
        for word in words:
            for c in word:
                char_counts[c] += 1
        
        total_pairs = 0
        for c, cnt in char_counts.items():
            total_pairs += cnt // 2
        
        # Calculate the required pairs for each word and sort them
        required_pairs = [len(word) // 2 for word in words]
        required_pairs.sort()
        
        current_sum = 0
        res = 0
        for pairs in required_pairs:
            if current_sum + pairs <= total_pairs:
                current_sum += pairs
                res += 1
            else:
                break
        
        return res