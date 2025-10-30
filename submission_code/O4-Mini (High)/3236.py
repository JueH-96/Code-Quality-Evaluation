from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # find the length of the longest sequential prefix
        n = len(nums)
        end = 1
        while end < n and nums[end] == nums[end - 1] + 1:
            end += 1
        
        # sum of that prefix
        prefix_sum = sum(nums[:end])
        
        # find the smallest x >= prefix_sum that's missing from nums
        seen = set(nums)
        x = prefix_sum
        while x in seen:
            x += 1
        return x