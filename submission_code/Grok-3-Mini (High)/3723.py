class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            if (i - k < 0 or nums[i] > nums[i - k]) and (i + k >= n or nums[i] > nums[i + k]):
                total += nums[i]
        return total