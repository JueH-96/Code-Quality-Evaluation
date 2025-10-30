class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        sum_squares = 0
        for i in range(n):
            position = i + 1
            if n % position == 0:
                sum_squares += nums[i] ** 2
        return sum_squares