from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If there is only one element, it is trivially special.
        if len(nums) <= 1:
            return True
        
        # Check each adjacent pair to see if they have different parity.
        for i in range(1, len(nums)):
            # If both numbers are even or both are odd, return False.
            if nums[i] % 2 == nums[i-1] % 2:
                return False
        
        return True