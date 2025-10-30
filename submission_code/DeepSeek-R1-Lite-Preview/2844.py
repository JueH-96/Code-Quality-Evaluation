class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        sum_squares = 0
        n = len(nums)
        for i in range(1, n + 1):
            if n % i == 0:
                sum_squares += nums[i - 1] ** 2
        return sum_squares