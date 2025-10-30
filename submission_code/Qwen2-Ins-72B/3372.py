class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        longest = 1
        increasing = [1] * n
        decreasing = [1] * n

        for i in range(1, n):
            if nums[i] > nums[i - 1]:
                increasing[i] = increasing[i - 1] + 1
            if nums[i] < nums[i - 1]:
                decreasing[i] = decreasing[i - 1] + 1

        longest = max(max(increasing), max(decreasing))
        return longest