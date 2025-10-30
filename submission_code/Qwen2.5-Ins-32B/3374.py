from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        start = 0
        for end in range(len(nums)):
            if end > 0 and nums[end] == nums[end - 1]:
                start = end
            count += end - start + 1
        return count