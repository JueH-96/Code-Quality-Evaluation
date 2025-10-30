from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for j in range(n):
            i = j + 1
            if n % i == 0:
                total += nums[j] ** 2
        return total