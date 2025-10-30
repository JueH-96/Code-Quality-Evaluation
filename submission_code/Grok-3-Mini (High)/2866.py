from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        for l in range(len(nums)):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_end = l
                current_length = 1
                while current_end + 1 < len(nums) and nums[current_end + 1] <= threshold and nums[current_end + 1] % 2 != nums[current_end] % 2:
                    current_end += 1
                    current_length += 1
                max_length = max(max_length, current_length)
        return max_length