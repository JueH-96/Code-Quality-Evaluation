class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        for i in range(len(nums)):
            if n % (i+1) == 0:
                result += nums[i] ** 2
        return result