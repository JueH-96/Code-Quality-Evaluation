class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        max_len = 1
        current_inc = 1
        current_dec = 1
        for i in range(1, len(nums)):
            prev_inc = current_inc
            prev_dec = current_dec
            if nums[i] > nums[i-1]:
                current_inc = prev_inc + 1
                current_dec = 1
            elif nums[i] < nums[i-1]:
                current_dec = prev_dec + 1
                current_inc = 1
            else:
                current_inc = 1
                current_dec = 1
            current_max = max(current_inc, current_dec)
            if current_max > max_len:
                max_len = current_max
        return max_len