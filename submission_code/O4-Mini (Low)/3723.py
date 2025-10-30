from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        
        for i, x in enumerate(nums):
            good = True
            # Check left neighbor at i - k
            if i - k >= 0 and x <= nums[i - k]:
                good = False
            # Check right neighbor at i + k
            if i + k < n and x <= nums[i + k]:
                good = False
            
            if good:
                total += x
        
        return total