from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        
        for i, num in enumerate(nums):
            if num > 0:
                landing_idx = (i + num) % n
                result.append(nums[landing_idx])
            elif num < 0:
                landing_idx = (i + num) % n
                result.append(nums[landing_idx])
            else:
                result.append(0)
        
        return result