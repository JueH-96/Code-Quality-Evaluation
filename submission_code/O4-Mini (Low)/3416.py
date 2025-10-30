from typing import List
from collections import Counter

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert numbers to strings (they all have the same length)
        strs = list(map(str, nums))
        n = len(strs)
        m = len(strs[0])
        
        # Precompute total number of pairs
        total_pairs = n * (n - 1) // 2
        result = 0
        
        # For each digit position, count the frequency of each digit
        for pos in range(m):
            freq = Counter(s[pos] for s in strs)
            # Count how many pairs have the same digit at this position
            same_pairs = sum(count * (count - 1) // 2 for count in freq.values())
            # Pairs that differ at this position
            diff_pairs = total_pairs - same_pairs
            result += diff_pairs
        
        return result