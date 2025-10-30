from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        
        for i in range(n):
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                current_length = 1
                for j in range(i + 1, n):
                    if nums[j] > threshold or nums[j] % 2 == nums[j - 1] % 2:
                        break
                    current_length += 1
                max_length = max(max_length, current_length)
        
        return max_length