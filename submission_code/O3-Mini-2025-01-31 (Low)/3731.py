from typing import List

class Solution:
    def subarraySum(self, nums: List[int]) -> int:
        total = 0
        for i in range(len(nums)):
            start = max(0, i - nums[i])
            # Sum subarray from `start` to `i` (inclusive)
            total += sum(nums[start:i+1])
        return total