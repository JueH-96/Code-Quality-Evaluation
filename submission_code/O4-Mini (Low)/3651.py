from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i, v in enumerate(nums):
            # Calculate the landing index with wrap around
            new_idx = (i + v) % n
            result[i] = nums[new_idx]
        
        return result