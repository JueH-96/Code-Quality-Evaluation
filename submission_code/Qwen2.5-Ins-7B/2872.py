class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        i = n - 2
        while i >= 0:
            if nums[i] <= nums[i + 1]:
                nums[i + 1] += nums[i]
            i -= 1
        return max(nums)