from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        """
        Determines if an array can be sorted by swapping adjacent elements
        with the same number of set bits.
        """
        n = len(nums)
        i = 0
        
        # Keep track of the maximum value of the previously processed group.
        # Since nums contains positive integers, 0 is a safe initial value.
        last_group_max = 0
        
        while i < n:
            # Determine the popcount for the current group and its initial min/max.
            # int.bit_count() is used for popcount (available in Python 3.10+).
            # For older Python versions, bin(nums[i]).count('1') can be used.
            p_count = nums[i].bit_count()
            current_min = nums[i]
            current_max = nums[i]
            
            i += 1
            
            # Extend the group as long as elements have the same popcount,
            # updating the group's min and max on the fly.
            while i < n and nums[i].bit_count() == p_count:
                current_min = min(current_min, nums[i])
                current_max = max(current_max, nums[i])
                i += 1
            
            # Check if the current sorted group can follow the previous one.
            # The smallest element of the current group must be at least as
            # large as the largest element of the previous group.
            if current_min < last_group_max:
                return False
                
            # Update the max for the next group's comparison.
            last_group_max = current_max
            
        return True