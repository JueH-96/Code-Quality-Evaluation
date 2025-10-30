from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        for i in range(n):
            start = max(0, i - nums[i])
            total_sum += sum(nums[start : i + 1])
        return total_sum