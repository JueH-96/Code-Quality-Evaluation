from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 0
        start = 0
        n = len(nums)
        for i in range(n):
            if i == start:
                current_and = nums[i]
            else:
                current_and &= nums[i]
            if current_and == 0:
                count += 1
                start = i + 1
        if start <= n - 1:
            count += 1
        return count