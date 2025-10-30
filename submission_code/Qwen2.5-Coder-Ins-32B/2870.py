from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = -1
        
        for i in range(n - 1):
            if nums[i + 1] == nums[i] + 1:
                length = 2
                j = i + 2
                while j < n and nums[j] == nums[j - 2]:
                    length += 1
                    j += 1
                max_length = max(max_length, length)
        
        return max_length