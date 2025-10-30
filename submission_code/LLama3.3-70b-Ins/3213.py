from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_val = max(nums)
        max_indices = [i for i, x in enumerate(nums) if x == max_val]
        
        count = 0
        for i in range(len(max_indices) - k + 1):
            left = max_indices[i]
            right = max_indices[i + k - 1]
            count += (left + 1) * (len(nums) - right)
        
        return count