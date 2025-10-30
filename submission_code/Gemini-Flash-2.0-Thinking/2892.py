from collections import Counter
from typing import List

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        if not nums:
            return False

        max_num = max(nums)
        n = max_num

        base_n = list(range(1, n)) + [n, n]

        if len(nums) != len(base_n):
            return False

        nums_counts = Counter(nums)
        base_n_counts = Counter(base_n)

        return nums_counts == base_n_counts