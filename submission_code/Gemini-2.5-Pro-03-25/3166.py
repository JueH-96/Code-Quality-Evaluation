import math
from collections import Counter
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum number of groups for a valid assignment.
    A valid assignment partitions indices [0, n-1] into groups such that:
    1. All indices `i` assigned to group `g` have the same value `nums[i]`.
    2. For any two groups `g_1` and `g_2`, the difference between the number of indices assigned
       to `g_1` and `g_2` should not exceed 1. This implies all group sizes must be `k` or `k+1`
       for some non-negative integer `k`. Let `s = k+1`. Then the allowed group sizes are `s` and `s-1`.
       Note that `s >= 1` since `k >= 0`.
    """
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of groups required for a valid assignment.

        Args:
            nums: A 0-indexed integer array of length n.

        Returns:
            An integer denoting the minimum number of groups needed.
        """
        
        # Step 1: Count the frequency of each distinct number in nums.
        # The indices corresponding to a specific value must form groups among themselves.
        counts = Counter(nums)
        
        # Extract the frequency values. Each value `c` represents the number of indices
        # that have a certain number, and these `c` indices must be partitioned.
        count_values = list(counts.values())
        
        # Constraints state 1 <= nums.length <= 10^5, so nums is non-empty,
        # and count_values will contain at least one element.
        
        # Step 2: Determine the search range for the parameter 's'.
        # Let 's' be the larger possible group size (k+1). Allowed group sizes are 's' and 's-1'.
        # For any count 'c', it must be possible to partition 'c' items into groups of size 's' or 's-1'.
        # This implies that 'c' must be representable as c = p*s + q*(s-1) for non-negative integers p, q.
        # The smallest possible group size is s-1 (if s > 1) or 1 (if s=1).
        # A count 'c' must accommodate at least one group. If s > 1, c >= s-1 must hold.
        # This must hold for all counts, including the minimum count `min_c`.
        # So, `min_c >= s-1`, which implies `s <= min_c + 1`.
        # This gives an upper bound for the search range of `s`.
        min_c = min(count_values)
        
        # Step 3: Find the largest valid 's' to minimize total groups.
        # The total number of groups for a valid 's' is the sum over all counts `c` of the minimum
        # number of groups needed to partition `c`. This minimum is ceil(c/s).
        # The total number of groups T(s) = sum(ceil(c_i / s)) is a non-increasing function of 's'.
        # Therefore, the largest valid 's' will yield the minimum total number of groups.
        # We iterate 's' downwards from the upper bound `min_c + 1` down to 1.
        
        for s in range(min_c + 1, 0, -1):
            # Assume the current 's' is valid until proven otherwise
            is_s_valid = True
            # Accumulate the total number of groups required if this 's' is valid
            total_groups_for_s = 0
            
            # Check if 's' is valid for all counts `c` in count_values
            for c in count_values:
                # Check if count 'c' can be partitioned into groups of size 's' and 's-1'.
                # This partition is possible if and only if there exists an integer T (number of groups for c)
                # such that ceil(c/s) <= T <= floor(c/(s-1)).
                # This requires the interval [ceil(c/s), floor(c/(s-1))] to contain at least one integer.
                # The condition for this is: ceil(c/s) <= floor(c/(s-1)).
                
                if s == 1:
                    # Base case: s=1 implies group sizes are 1 and 0. Since group size must be positive,
                    # only groups of size 1 are practical. Any count `c` can be partitioned into `c` groups of size 1.
                    # This is always possible. The number of groups required for this count `c` is `c`.
                    num_groups_for_c = c
                else: # s >= 2
                    # Calculate ceil(c/s) using integer arithmetic.
                    # For positive integers a, b: ceil(a/b) = (a + b - 1) // b
                    ceil_div = (c + s - 1) // s
                    
                    # Calculate floor(c/(s-1)). Since s >= 2, s-1 >= 1, so division by zero is not possible.
                    floor_div = c // (s - 1) 
                    
                    # Check the validity condition
                    if ceil_div > floor_div:
                        # If the condition fails for any count `c`, then this 's' is invalid.
                        is_s_valid = False
                        # No need to check other counts for this 's'; break and try the next smaller 's'.
                        break 
                    
                    # If the partition is possible for this 'c', the minimum number of groups
                    # required for this 'c' (given 's') is ceil(c/s).
                    num_groups_for_c = ceil_div
                
                # Add the number of groups for this count 'c' to the total sum for the current 's'
                total_groups_for_s += num_groups_for_c

            # After checking all counts for the current 's'
            if is_s_valid:
                # If 's' is valid for all counts, we have found the largest valid 's'
                # because we are iterating downwards. This 's' gives the minimum total groups.
                return total_groups_for_s
        
        # Fallback return statement. According to the logic, s=1 is always a valid configuration
        # (partitioning each count `c` into `c` groups of size 1, total N groups).
        # Thus, the loop should always find a valid `s` (at least s=1) and return a value.
        # This line should theoretically be unreachable.
        # Returning len(nums) as it corresponds to the result for s=1.
        return len(nums) # Should be unreachable based on analysis.