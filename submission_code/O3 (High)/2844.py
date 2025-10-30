from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        """
        For every index i (1-indexed) that divides the array length n,
        add the square of nums[i-1] (0-indexed access) to the answer.
        """
        n = len(nums)
        ans = 0
        for i in range(1, n + 1):          # 1-indexed positions
            if n % i == 0:                 # i is a divisor of n → special element
                ans += nums[i - 1] ** 2
        return ans