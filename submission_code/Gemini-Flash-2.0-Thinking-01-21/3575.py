from typing import List
from itertools import combinations

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_val = 0
        indices_combinations = combinations(range(n), 2 * k)
        for indices in indices_combinations:
            subsequence = [nums[i] for i in indices]
            or1 = 0
            for i in range(k):
                or1 |= subsequence[i]
            or2 = 0
            for i in range(k, 2 * k):
                or2 |= subsequence[i]
            current_val = or1 ^ or2
            max_val = max(max_val, current_val)
        return max_val