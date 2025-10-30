from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        # nums is 1-indexed conceptually, so nums[i] is at index i-1
        for i in range(1, n+1):
            if n % i == 0:
                total += nums[i-1] * nums[i-1]
        return total