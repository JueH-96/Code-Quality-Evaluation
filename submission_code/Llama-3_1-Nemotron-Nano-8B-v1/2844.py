from typing import List

class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        sum_squares = 0
        for j in range(n):
            i = j + 1
            if n % i == 0:
                sum_squares += nums[j] ** 2
        return sum_squares