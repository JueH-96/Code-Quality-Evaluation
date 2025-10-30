class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        special_elements_sum = 0
        for i in range(1, n + 1):
            if n % i == 0:
                special_elements_sum += nums[i - 1] ** 2
        return special_elements_sum