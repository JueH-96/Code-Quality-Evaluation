from typing import List
from itertools import combinations

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_value = 0

        for subseq_indices in combinations(range(n), 2 * k):
            subseq_elements = [nums[i] for i in sorted(subseq_indices)]

            first_half_or = 0
            for i in range(k):
                first_half_or |= subseq_elements[i]

            second_half_or = 0
            for i in range(k, 2 * k):
                second_half_or |= subseq_elements[i]

            current_value = first_half_or ^ second_half_or
            max_value = max(max_value, current_value)

        return max_value