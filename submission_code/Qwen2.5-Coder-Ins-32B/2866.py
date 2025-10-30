from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        
        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                length = 1
                j = i + 1
                while j < n and nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                    length += 1
                    j += 1
                max_length = max(max_length, length)
        
        return max_length