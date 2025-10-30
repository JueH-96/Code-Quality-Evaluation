from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                r = l + 1
                while r < n and nums[r] <= threshold and nums[r] % 2 != nums[r - 1] % 2:
                    r += 1
                max_length = max(max_length, r - l)
        
        return max_length