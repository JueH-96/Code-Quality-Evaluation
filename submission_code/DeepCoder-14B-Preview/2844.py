class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for j in range(n):
            if n % (j + 1) == 0:
                total += nums[j] ** 2
        return total