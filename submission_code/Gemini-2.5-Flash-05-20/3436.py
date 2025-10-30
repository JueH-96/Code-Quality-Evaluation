import math
from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # Initialize min_diff to a very large number (infinity)
        min_diff = float('inf')
        
        # current_ors will store all distinct bitwise OR values of subarrays
        # that end at the current index being processed.
        # The key optimization relies on the fact that the number of distinct
        # OR values for subarrays ending at any specific index is at most ~30
        # (the number of bits in the maximum possible value, 2^30 for 10^9).
        current_ors = set() 
        
        for num in nums:
            # Create a new set to store OR values for subarrays ending at the current 'num'.
            next_ors = set()
            
            # Case 1: The subarray consists only of the current 'num' itself (e.g., nums[i..i]).
            next_ors.add(num)
            
            # Case 2: Extend subarrays that ended at the previous index.
            # For every OR value 'prev_or_val' that ended at the previous index,
            # 'prev_or_val | num' forms a new OR value ending at the current index.
            for prev_or_val in current_ors:
                next_ors.add(prev_or_val | num)
            
            # Update current_ors to reflect the new set of OR values ending at the current index.
            current_ors = next_ors
            
            # After computing all possible OR values ending at the current index,
            # check them against 'k' to find the minimum absolute difference.
            for or_val in current_ors:
                min_diff = min(min_diff, abs(k - or_val))
                
                # Optimization: If the difference is 0, we found an exact match,
                # which is the smallest possible difference, so we can return immediately.
                if min_diff == 0:
                    return 0
                    
        # Return the minimum difference found across all subarrays.
        return min_diff