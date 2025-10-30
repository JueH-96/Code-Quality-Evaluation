from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        current_length = 1
        count += current_length
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current_length += 1
            else:
                current_length = 1
            count += current_length
        return count