class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        sum_squares = 0
        for i in range(1, n + 1):
            if n % i == 0:
                sum_squares += nums[i - 1] ** 2
        return sum_squares