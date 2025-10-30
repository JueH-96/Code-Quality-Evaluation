from typing import List
from itertools import combinations

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        max_value = 0
        for subseq in combinations(nums, 2 * k):
            first_half_or = 0
            second_half_or = 0
            for i in range(k):
                first_half_or |= subseq[i]
                second_half_or |= subseq[i + k]
            max_value = max(max_value, first_half_or ^ second_half_or)
        return max_value