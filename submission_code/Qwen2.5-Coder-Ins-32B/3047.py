from typing import List
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(x):
            s = int(math.sqrt(x))
            return s * s == x

        n = len(nums)
        max_sum = 0

        for i in range(1, n + 1):
            current_sum = 0
            j = 1
            while i * j * j <= n:
                current_sum += nums[i * j * j - 1]
                j += 1
            max_sum = max(max_sum, current_sum)

        return max_sum