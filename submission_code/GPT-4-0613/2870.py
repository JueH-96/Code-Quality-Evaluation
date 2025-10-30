class Solution:
    def alternatingSubarray(self, nums):
        max_length = 1
        current_length = 1
        for i in range(1, len(nums)):
            if (nums[i] - nums[i - 1]) * (-1) ** (i - 1) == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        return max_length if max_length > 1 else -1