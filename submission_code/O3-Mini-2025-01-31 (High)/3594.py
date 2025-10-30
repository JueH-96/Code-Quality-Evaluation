from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total = sum(nums)
        freq = Counter(nums)
        ans = None
        
        # A candidate x can serve as the outlier if:
        # (total - x) is even (so that S = (total-x)//2 is an integer),
        # and there is at least one occurrence of S in nums.
        # If x equals S, then there must be at least 2 occurrences to use two different indices.
        for x in freq:
            if (total - x) % 2 != 0:
                continue
            S = (total - x) // 2
            if S not in freq:
                continue
            if x == S and freq[x] < 2:
                continue
            if ans is None or x > ans:
                ans = x
        return ans