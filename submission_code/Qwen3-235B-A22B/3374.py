from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        total = 1
        current = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current += 1
            else:
                current = 1
            total += current
        return total