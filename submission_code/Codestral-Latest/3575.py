from itertools import combinations
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        def calculate_value(seq):
            x = len(seq) // 2
            first_part = 0
            second_part = 0
            for i in range(x):
                first_part |= seq[i]
                second_part |= seq[x + i]
            return first_part ^ second_part

        max_val = 0
        for seq in combinations(nums, 2 * k):
            max_val = max(max_val, calculate_value(seq))

        return max_val