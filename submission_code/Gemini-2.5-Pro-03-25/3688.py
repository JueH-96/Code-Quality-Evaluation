import collections
from typing import List
import math

# Using a large negative number to represent -infinity for integer arithmetic.
# The constraints state -10^6 <= nums[i] <= 10^6 and 1 <= nums.length <= 10^5.
# Maximum possible sum is 10^5 * 10^6 = 10^11.
# Minimum possible sum is 10^5 * -10^6 = -10^11.
# A value safely smaller than any possible sum, e.g., -10^12 or larger negative.
# Using -10**18 for absolute safety. Python's integers handle arbitrary precision.
INT_MIN_SENTINEL = -10**18 

class Solution:
    
    def _kadane_int(self, arr: List[int]) -> int:
        """ 
        Helper function implementing Kadane's algorithm to find the maximum subarray sum.
        Uses integer arithmetic suitable for Python's arbitrary precision integers.
        Handles edge cases like all negative numbers.
        Assumes arr is non-empty based on problem constraints and logic.
        """
        if not arr:
            # This path should logically not be reached given the problem constraints (N >= 1)
            # and the guarantee that the array remains non-empty after the operation.
            # Return the sentinel value indicating an error or unexpected empty array.
            return INT_MIN_SENTINEL 
        
        # Initialize max_so_far and current_max with the first element.
        # This correctly handles arrays starting with negative numbers or single-element arrays.
        max_so_far = arr[0]
        current_max = arr[0]
        
        # Iterate through the array starting from the second element.
        for i in range(1, len(arr)):
            val = arr[i]
            # Kadane's core logic: update the maximum sum ending at the current position.
            # It's either the current element itself or the current element added to the previous max sum ending there.
            current_max = max(val, current_max + val)
            # Update the overall maximum sum found so far across all positions.
            max_so_far = max(max_so_far, current_max)
            
        return max_so_far

    def maxSubarraySum(self, nums: List[int]) -> int:
        """
        Finds the maximum subarray sum across all possible arrays resulting from 
        at most one operation of removing all occurrences of an integer x.
        The operation is valid only if the array remains non-empty after removal.
        """
        N = len(nums)
        # Problem constraints state 1 <= N <= 10^5, so N=0 check is technically not needed.
        
        # Calculate the maximum subarray sum for the original array using Kadane's algorithm.
        # This covers the scenario where no operation is performed or is optimal.
        # Initialize the overall maximum sum found so far with this value.
        overall_max_sum = self._kadane_int(nums)

        # Count occurrences of each number. This is needed to:
        # 1. Identify the distinct elements to consider for removal.
        # 2. Check the validity condition `counts[x] < N` for removal.
        counts = collections.Counter(nums)
        # Get the list of distinct elements present in the array.
        distinct_elements = list(counts.keys())
        
        # Dictionaries to store the state of Kadane's algorithm for each scenario
        # where a distinct element `x` is hypothetically removed.
        # We simulate Kadane's for each `x` simultaneously.
        # Initialize states with the sentinel value representing negative infinity.
        current_max_dict = {x: INT_MIN_SENTINEL for x in distinct_elements}
        max_so_far_dict = {x: INT_MIN_SENTINEL for x in distinct_elements}
        # Flags to track if we've encountered the first valid element for each scenario `x`.
        # This is crucial for correct initialization of Kadane's state for each simulation path.
        found_first_dict = {x: False for x in distinct_elements}

        # Iterate through the original array nums element by element.
        for val in nums:
            # For each distinct element `x` that could potentially be removed:
            for x in distinct_elements:
                # If the current element `val` is NOT the element `x` being considered for removal:
                if val != x: 
                    # This means `val` would be part of the array in the scenario where `x` is removed.
                    # We process this element `val` using Kadane's logic for this specific scenario.
                    
                    # Use `x` itself (the integer value) as the key for the dictionaries.
                    x_key = x 
                    
                    if not found_first_dict[x_key]:
                        # This is the first non-`x` element encountered for this scenario.
                        # Initialize Kadane's state (`current_max` and `max_so_far`) for this scenario with `val`.
                        current_max_dict[x_key] = val
                        max_so_far_dict[x_key] = val
                        found_first_dict[x_key] = True
                    else:
                        # This is not the first non-`x` element. Apply standard Kadane's update rules.
                        # Update the maximum sum ending at the current position for the scenario removing `x`.
                        current_max_dict[x_key] = max(val, current_max_dict[x_key] + val)
                        # Update the overall maximum sum found so far for the scenario removing `x`.
                        max_so_far_dict[x_key] = max(max_so_far_dict[x_key], current_max_dict[x_key])

        # After iterating through all elements of nums:
        # `max_so_far_dict` contains the maximum subarray sum for each scenario where a distinct element `x` was removed.
        # Now, consolidate the results. Update the `overall_max_sum` with the maximum found among all valid removal scenarios.
        for x in distinct_elements:
            # A removal operation for element `x` is valid only if the array remains non-empty after removal.
            # This is true if the count of `x` is less than the total number of elements `N`.
            if counts[x] < N:
                # Additionally, we must ensure that the resulting array after removing `x` was indeed non-empty.
                # The `found_first_dict[x]` flag confirms that at least one non-`x` element was encountered and processed.
                if found_first_dict[x]:
                   # If the scenario is valid and produced a result, update the overall maximum sum.
                   overall_max_sum = max(overall_max_sum, max_so_far_dict[x])
            # If counts[x] == N, removing x would make the array empty, this operation is disallowed by the problem statement, so we ignore its result.
        
        # Return the final maximum subarray sum found across the original array and all valid removal scenarios.
        # The result is guaranteed to be an integer within the possible range of sums.
        return overall_max_sum