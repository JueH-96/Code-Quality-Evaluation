from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        total = 0
        curr_length = 0
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i - 1]:
                curr_length += 1  # Extend the alternating subarray
            else:
                curr_length = 1   # Reset when two consecutive elements are the same
            total += curr_length
        return total