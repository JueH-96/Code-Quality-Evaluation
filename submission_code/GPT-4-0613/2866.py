class Solution:
    def longestAlternatingSubarray(self, nums, threshold):
        max_len = 0
        for i in range(len(nums)):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                j = i + 1
                while j < len(nums) and nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                    j += 1
                max_len = max(max_len, j - i)
        return max_len