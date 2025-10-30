class Solution:
    def countAlternatingSubarrays(self, nums):
        count = 0
        left = 0
        for right in range(1, len(nums)):
            if nums[right] == nums[right - 1]:
                count += (right - left) * (right - left + 1) // 2
                left = right
        count += (len(nums) - left) * (len(nums) - left + 1) // 2
        return count