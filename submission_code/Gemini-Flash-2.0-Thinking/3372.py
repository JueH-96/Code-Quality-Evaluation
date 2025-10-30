class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n

        max_length = 1
        current_increasing_length = 1
        current_decreasing_length = 1

        for i in range(1, n):
            if nums[i] > nums[i-1]:
                current_increasing_length += 1
                current_decreasing_length = 1
            elif nums[i] < nums[i-1]:
                current_decreasing_length += 1
                current_increasing_length = 1
            else:
                current_increasing_length = 1
                current_decreasing_length = 1

            max_length = max(max_length, current_increasing_length, current_decreasing_length)

        return max_length