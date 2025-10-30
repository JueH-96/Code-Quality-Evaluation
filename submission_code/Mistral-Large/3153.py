from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7

        # Sort the array in descending order
        nums.sort(reverse=True)

        # Calculate the sum of squares of the largest k elements
        max_sum_of_squares = sum(x**2 for x in nums[:k]) % MOD

        return max_sum_of_squares