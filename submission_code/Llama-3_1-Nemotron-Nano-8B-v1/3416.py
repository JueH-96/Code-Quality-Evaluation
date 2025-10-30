from typing import List
from collections import defaultdict

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        num_str = str(nums[0])
        digits = len(num_str)
        total = 0
        for i in range(digits):
            power = 10 ** (digits - 1 - i)
            count = defaultdict(int)
            for num in nums:
                d = (num // power) % 10
                count[d] += 1
            sum_sq = sum(v * v for v in count.values())
            contribution = (n * n - sum_sq) // 2
            total += contribution
        return total