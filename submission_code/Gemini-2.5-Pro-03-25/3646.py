import sys 
# Setting recursion depth is not needed for this iterative solution.
# sys.setrecursionlimit(2000) 
from typing import List

class Solution:
  """
  This class provides a method to calculate the sum of all good subsequences
  of a given list of numbers `nums`. A good subsequence is defined as one
  where the absolute difference between any two consecutive elements is exactly 1.
  The sum is returned modulo 10^9 + 7.
  """
  def sumOfGoodSubsequences(self, nums: List[int]) -> int:
    """
    Calculates the sum of all good subsequences of nums modulo 10^9 + 7 using Dynamic Programming.

    The approach uses DP states `total_count[v]` and `total_sum[v]` which store the count
    and total sum of elements respectively for all good subsequences ending with value `v`
    encountered so far while iterating through `nums`.

    Args:
      nums: A list of integers. Constraints: 1 <= nums.length <= 10^5, 0 <= nums[i] <= 10^5.

    Returns:
      The sum of elements across all possible good subsequences, modulo 10^9 + 7.
    """
    MOD = 10**9 + 7

    # Handle the edge case of an empty input list. An empty list has no subsequences, sum is 0.
    if not nums:
      return 0

    # Determine the maximum value in nums to set the size of DP arrays.
    # This is needed because DP states are indexed by element values.
    max_val = 0
    # Using a loop to find max_val, this avoids issues if nums is very large, though max() works too.
    for x_val in nums:
        if x_val > max_val:
            max_val = x_val
    # Alternatively: max_val = max(nums) # This is safe due to the `if not nums` check above.

    # Initialize DP arrays.
    # Size needs to accommodate indices up to max_val + 1 (for accessing x+1 when x=max_val).
    # Python list indices go from 0 to length-1. So array length must be at least max_val + 2.
    size = max_val + 2
    
    # total_count[v]: stores the total count of good subsequences ending with value v encountered so far.
    total_count = [0] * size 
    # total_sum[v]: stores the sum of elements of all good subsequences ending with value v encountered so far.
    total_sum = [0] * size
    
    # final_total_sum accumulates the sum of elements over all good subsequences found.
    final_total_sum = 0

    # Iterate through each number in the input list.
    for x in nums:
        # For each number `x`, calculate the count and sum of good subsequences ending with `x` at this specific position in `nums`.
        
        # Base case: The subsequence consisting of only `x`.
        # The count of such a subsequence is 1.
        current_count = 1
        # The sum of elements in this subsequence is just `x`.
        # Since max x = 10^5, which is less than MOD, x % MOD is just x. Initialize with x.
        current_sum = x 
        
        # Extend good subsequences ending with `x-1`.
        # This is only possible if `x > 0`.
        if x > 0:
            # Retrieve the count and sum information for subsequences ending in x-1 processed so far.
            prev_count1 = total_count[x-1]
            prev_sum1 = total_sum[x-1]
            
            # Add the count of subsequences ending in `x-1`. Each can be extended by appending `x`.
            current_count = (current_count + prev_count1) % MOD
            
            # Calculate the sum contribution from extending these `prev_count1` subsequences.
            # The total sum contribution is (sum of elements in previous subsequences) + (count of previous subsequences * current value x).
            # Apply modulo arithmetic at each step.
            term_sum1 = (prev_sum1 + (prev_count1 * x) % MOD) % MOD
            # Add this contribution to `current_sum`.
            current_sum = (current_sum + term_sum1) % MOD

        # Extend good subsequences ending with `x+1`.
        # The index `x+1` is safe to access because `size = max_val + 2` and `x <= max_val`.
        # Thus, `x+1 <= max_val + 1`, which is a valid index.
        
        # Retrieve the count and sum information for subsequences ending in x+1 processed so far.
        prev_count2 = total_count[x+1]
        prev_sum2 = total_sum[x+1]

        # Add the count of subsequences ending in `x+1`. Each can be extended by appending `x`.
        current_count = (current_count + prev_count2) % MOD
        
        # Calculate sum contribution from extending these `prev_count2` subsequences.
        term_sum2 = (prev_sum2 + (prev_count2 * x) % MOD) % MOD
        # Add this contribution to `current_sum`.
        current_sum = (current_sum + term_sum2) % MOD
            
        # Update the DP state for value `x`. This aggregates the information derived from processing `x` at the current position.
        # The updated state `total_count[x]` and `total_sum[x]` includes the `current_count` and `current_sum`.
        # This updated state will be available for subsequent elements in `nums`.
        total_count[x] = (total_count[x] + current_count) % MOD
        total_sum[x] = (total_sum[x] + current_sum) % MOD
        
        # Add the `current_sum` (which is the total sum of elements for all good subsequences ending specifically with `x` at this position)
        # to the overall total sum `final_total_sum`.
        final_total_sum = (final_total_sum + current_sum) % MOD

    # Return the final accumulated sum modulo MOD.
    return final_total_sum