from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0

        max_len = 1
        current_len_inc = 1
        current_len_dec = 1

        for i in range(1, nums.length):
            if nums[i] > nums[i - 1]:
                current_len_inc += 1
                current_len_dec = 1  # Reset decreasing count
            elif nums[i] < nums[i - 1]:
                current_len_dec += 1
                current_len_inc = 1  # Reset increasing count
            else:
                current_len_inc = 1
                current_len_dec = 1

            max_len = max(max_len, current_len_inc, current_len_dec)

        return max_len