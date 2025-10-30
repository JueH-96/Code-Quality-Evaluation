from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        max_a = n - 2 * k
        
        for a in range(max_a + 1):
            # Check if the first subarray starting at a is strictly increasing
            valid1 = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid1 = False
                    break
            if not valid1:
                continue
            
            # Check if the second subarray starting at a + k is strictly increasing
            b = a + k
            valid2 = True
            for i in range(b, b + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid2 = False
                    break
            
            if valid2:
                return True
        
        return False