from typing import List
from collections import Counter
import math

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = Counter(nums)
        unique_nums = set(nums)
        
        maxLength = 1 # Minimum possible length is 1 as nums.length >= 2
        
        # Handle base 1 separately
        if counts[1] > 0:
            # The pattern is [1, 1, ..., 1] of length 2m+1.
            # Requires 2m+1 ones. Max length is the largest odd number <= counts[1].
            # If counts[1] = k >= 1, max length is k if k is odd, k-1 if k is even.
            # This is counts[1] - (counts[1] % 2 == 0).
            max_len_base_1 = counts[1] - (counts[1] % 2 == 0)
            maxLength = max(maxLength, max_len_base_1)

        # Handle bases > 1
        # Iterate through unique numbers present in counts.
        # Sorting unique numbers is not strictly necessary but could help structure the checks.
        # for n in sorted(counts): 
        for n in counts:
            if n == 1:
                continue # Handled already

            # Check if n is a perfect square of a number already in nums (and that number is > 1)
            # If n = y*y where y is in unique_nums and y > 1, skip n as a base.
            is_square_of_other_in_nums = False
            if n > 1: # Perfect squares <= 1 are only 1, handled already.
                sqrt_n_float = math.sqrt(n)
                sqrt_n_int = round(sqrt_n_float)
                
                # Check if sqrt_n_int is the exact integer square root
                if sqrt_n_int * sqrt_n_int == n:
                    # If sqrt_n_int > 1, check if it's in unique_nums.
                    if sqrt_n_int > 1 and sqrt_n_int in unique_nums:
                         is_square_of_other_in_nums = True

            if is_square_of_other_in_nums:
                # Skip this number as a base, as a potentially longer sequence starts from its square root.
                continue

            # This number n is a potential base x (v_0) for a new sequence chain.
            
            possible_levels = []
            curr = n
            
            # Build the core sequence [v_0, v_1, v_2, ...] as long as elements are available in counts
            # and do not exceed 10^9.
            
            while curr <= 10**9 and counts[curr] > 0:
                 possible_levels.append(curr)
                 
                 # Prepare for next power, check for potential overflow *before* multiplication.
                 # If curr > sqrt(10^9), then curr * curr will be > 10^9.
                 # Use integer division for safety: if curr > 10**9 // curr, then curr * curr > 10^9.
                 # Note: this check requires curr >= 1. Since n > 1 here, curr starts >= 2.
                 if curr > 10**9 // curr:
                     break # Next value will exceed 10^9
                     
                 curr = curr * curr
            
            # Now possible_levels = [v_0, v_1, ..., v_p] where v_i = n^(2^i) and counts[v_i] > 0
            # We can form patterns of length 2m + 1 for 0 <= m <= p.
            # Pattern length 2m + 1 requires counts[v_i] >= 2 for 0 <= i < m, and counts[v_m] >= 1.
            
            # Iterate through possible m values (level of the central element)
            # m goes from 0 up to p = len(possible_levels) - 1.
            
            for m in range(len(possible_levels)): # m corresponds to the index in possible_levels
                 # Check if we have enough counts for pattern of length 2m+1
                 # Requirements: counts[possible_levels[i]] >= 2 for 0 <= i < m, and counts[possible_levels[m]] >= 1.
                 
                 can_form_m = True
                 
                 # Check symmetric elements count requirement (needs 2 each) for indices 0 up to m-1
                 for i in range(m):
                     if counts[possible_levels[i]] < 2:
                         can_form_m = False
                         break
                 
                 # Check central element count requirement (needs 1) for index m
                 # Only need to check if the previous loop didn't fail.
                 if can_form_m and counts[possible_levels[m]] >= 1:
                     # Possible to form pattern of length 2m+1
                     current_pattern_length = 2 * m + 1
                     maxLength = max(maxLength, current_pattern_length)
                 else:
                     # If we cannot form pattern of length 2m+1, we cannot form longer patterns 2(m+k)+1 (k>0) either.
                     break # Exit inner loop over m

        return maxLength