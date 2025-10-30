from typing import List

class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        i = 0
        
        while i < n:
            count += 1
            j = i + 1
            while j < n and nums[j] != nums[j - 1]:
                count += 1
                j += 1
            i = j
        
        return count