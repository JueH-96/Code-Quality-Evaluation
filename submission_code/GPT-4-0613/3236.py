class Solution:
    def missingInteger(self, nums: list[int]) -> int:
        prefix_sum = nums[0]
        i = 1
        while i < len(nums) and nums[i] == nums[i - 1] + 1:
            prefix_sum += nums[i]
            i += 1
        x = prefix_sum
        while x in nums:
            x += 1
        return x