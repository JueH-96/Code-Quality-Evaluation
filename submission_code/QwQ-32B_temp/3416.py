from typing import List
from collections import Counter

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        digits_list = list(map(str, nums))
        k = len(digits_list[0])
        n = len(nums)
        total = 0
        
        for pos in range(k):
            digits = [s[pos] for s in digits_list]
            cnt = Counter(digits)
            same_pairs = sum(c * (c - 1) // 2 for c in cnt.values())
            total_pairs = n * (n - 1) // 2
            total += (total_pairs - same_pairs)
        
        return total