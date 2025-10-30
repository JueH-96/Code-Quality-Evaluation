from functools import reduce
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        def get_or_xor_value(start, end):
            left_or = reduce(lambda x, y: x | y, nums[start:start + k], 0)
            right_or = reduce(lambda x, y: x | y, nums[start + k:end], 0)
            return left_or ^ right_or

        n = len(nums)
        max_value = 0
        for i in range(n - 2*k + 1):
            max_value = max(max_value, get_or_xor_value(i, i + 2*k))
        return max_value