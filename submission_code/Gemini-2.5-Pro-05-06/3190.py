import math
from typing import List

class Solution:
  def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
    n = len(nums1)
    
    # Use float('inf') for infinity. This is a common practice.
    infinity = math.inf

    # Helper function to calculate the minimum swaps needed for the prefix
    # (elements from index 0 to n-2) to satisfy the conditions,
    # given the target maximum values for nums1 and nums2.
    # These target maximums are determined by the chosen values at index n-1.
    # The function returns the number of swaps, or infinity if impossible.
    def calculate_prefix_swaps(target_max1: int, target_max2: int) -> float:
        swaps_count = 0
        # Iterate through elements from index 0 to n-2.
        # If n=1, n-1=0, so range(0) is empty, and the loop doesn't run.
        # In this case, swaps_count remains 0, which is correct.
        for i in range(n - 1): 
            val1_i = nums1[i]  # Current value nums1[i]
            val2_i = nums2[i]  # Current value nums2[i]
            
            # Option 1: Keep nums1[i] and nums2[i] as they are.
            # The elements at index i would be (val1_i, val2_i).
            # This is possible if val1_i <= target_max1 and val2_i <= target_max2.
            keep_possible = (val1_i <= target_max1 and val2_i <= target_max2)
            
            # Option 2: Swap nums1[i] and nums2[i].
            # The elements at index i would be (val2_i, val1_i).
            # This is possible if val2_i <= target_max1 and val1_i <= target_max2.
            swap_possible = (val2_i <= target_max1 and val1_i <= target_max2)
            
            if keep_possible:
                # If keeping elements as they are is possible, this costs 0 swaps for index i.
                # This is preferred even if swapping is also possible (0 swaps vs 1 swap).
                pass
            elif swap_possible:
                # If keeping is not possible, but swapping is, we must swap.
                # This costs 1 swap for index i.
                swaps_count += 1
            else:
                # Neither keeping nor swapping works to satisfy the conditions for this index i
                # with the given target_max1 and target_max2.
                # So, this path (choice of values at n-1) is impossible.
                return infinity
        
        # If the loop completes, all prefix elements can satisfy the conditions.
        # Return the total swaps for the prefix. Can be float due to infinity.
        return float(swaps_count)

    # --- Scenario 1: Do NOT swap elements at index n-1 ---
    # In this scenario, final nums1[n-1] is original nums1[n-1],
    # and final nums2[n-1] is original nums2[n-1].
    # These values become the target maximums for their respective arrays.
    s1_target_max1 = nums1[n-1]
    s1_target_max2 = nums2[n-1]
    
    # Calculate swaps needed for the prefix elements (indices 0 to n-2).
    s1_prefix_swaps = calculate_prefix_swaps(s1_target_max1, s1_target_max2)
    
    # Total cost for Scenario 1 = (swaps for prefix) + (0 swaps for index n-1).
    cost1 = s1_prefix_swaps 
    
    # --- Scenario 2: Swap elements at index n-1 ---
    # In this scenario, final nums1[n-1] is original nums2[n-1],
    # and final nums2[n-1] is original nums1[n-1].
    # This operation itself costs 1 swap.
    # These swapped values become the target maximums.
    s2_target_max1 = nums2[n-1] # Original nums2[n-1] now in nums1[n-1]
    s2_target_max2 = nums1[n-1] # Original nums1[n-1] now in nums2[n-1]
    
    # Calculate swaps needed for the prefix elements (indices 0 to n-2).
    s2_prefix_swaps = calculate_prefix_swaps(s2_target_max1, s2_target_max2)
    
    # Initialize cost2 as infinity. It will be updated if this scenario is possible.
    cost2 = infinity 
    if s2_prefix_swaps != infinity:
        # Total cost for Scenario 2 = (swaps for prefix) + (1 swap for index n-1).
        cost2 = s2_prefix_swaps + 1.0 # Add 1.0 to keep type as float
            
    # The overall minimum operations is the minimum of costs from the two scenarios.
    min_total_swaps = min(cost1, cost2)
    
    # If min_total_swaps is still infinity, it means neither scenario leads to a solution.
    if min_total_swaps == infinity:
        return -1
    else:
        # Otherwise, min_total_swaps is the minimum number of operations (it will be finite).
        # The problem requires an integer return type.
        return int(min_total_swaps)