from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total = 0
        for i in range(len(nums)):
            start = max(0, i - nums[i])
            current_sum = 0
            for j in range(start, i + 1):
                current_sum += nums[j]
            total += current_sum
        return total