class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        total = 0
        for idx in range(n):
            if n % (idx + 1) == 0:
                total += nums[idx] ** 2
        return total