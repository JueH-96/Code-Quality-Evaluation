import math
from typing import List 

class Solution:
  """
  Finds a subarray of nums such that the absolute difference between k 
  and the bitwise OR of the subarray elements is minimized.
  """
  def minimumDifference(self, nums: List[int], k: int) -> int:
    """
    Calculates the minimum possible absolute difference |k - OR(subarray)|.

    The approach uses dynamic programming based on the observation that the set of 
    distinct bitwise OR values of subarrays ending at a particular index `i` is small.
    Let `S_i` be the set of distinct OR values of all subarrays `nums[l..i]` for `0 <= l <= i`.
    We can compute `S_i` based on `S_{i-1}` using the relation:
    `S_i = { nums[i] } U { val | nums[i] | val in S_{i-1} }`.
    The size of `S_i` is bounded by the number of bits in the integers (approx. 30), because
    the sequence `OR(i, i), OR(i-1, i), ..., OR(0, i)` is non-decreasing, and each increase
    requires at least one new bit to be set.

    Args:
        nums: A list of integers.
        k: An integer target value.

    Returns:
        The minimum absolute difference found.
    """
    
    # Initialize minimum difference to positive infinity.
    # Using float('inf') is standard practice for minimization problems.
    min_diff = float('inf')
    
    # `previous_S` stores the distinct OR values of all subarrays ending at the previous index (i-1).
    # Initially, this set is empty as there's no index before 0.
    previous_S = set()
    
    # Iterate through the array elements one by one.
    for i in range(len(nums)):
        num = nums[i]
        
        # `current_S` will store the distinct OR values of all subarrays ending at the current index `i`.
        current_S = set()
        
        # Consider the subarray containing only the current element `nums[i]`.
        # Its OR value is simply `num`. This is one possible OR value for a subarray ending at `i`.
        current_S.add(num)
        # Update the overall minimum difference found so far using this single-element subarray's OR value.
        min_diff = min(min_diff, abs(k - num))
        
        # Optimization: If the minimum difference is already 0, we have found a subarray whose OR is exactly k.
        # We cannot achieve a smaller absolute difference, so we can return early.
        if min_diff == 0:
            return 0
            
        # Now, consider extending the subarrays that ended at index `i-1`.
        # For each distinct OR value `val` present in `previous_S` (representing OR of some subarray `nums[l..i-1]`),
        # the OR value of the extended subarray `nums[l..i]` is `val | num`.
        for val in previous_S:
            new_val = val | num
            
            # We only need to consider `new_val` if it's a distinct OR value not already processed for index `i`.
            # If `new_val` is already in `current_S`, it means this OR value has already been considered
            # (either it equals `num`, or it was generated from a different `val'` in `previous_S`).
            # Checking this prevents redundant `min` calculations.
            if new_val not in current_S: 
               current_S.add(new_val)
               # Update the overall minimum difference with the absolute difference for this `new_val`.
               min_diff = min(min_diff, abs(k - new_val))
            
            # Another check for the early exit optimization. If after updating `min_diff` it becomes 0, return immediately.
            if min_diff == 0:
                return 0

        # After iterating through all values in `previous_S` and potentially adding new OR values,
        # `current_S` now holds all distinct OR values for subarrays ending at index `i`.
        # This set becomes the `previous_S` for the next iteration (index i+1).
        previous_S = current_S
            
    # After iterating through all elements of the array, `min_diff` will hold the minimum absolute difference found across all subarrays.
    return min_diff