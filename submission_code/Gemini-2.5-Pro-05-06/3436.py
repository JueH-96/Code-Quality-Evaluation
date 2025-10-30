import math
from typing import List

class Solution:
  def minimumDifference(self, nums: List[int], k: int) -> int:
    min_abs_difference = math.inf
    
    # q_prev stores the set of distinct OR values of subarrays ending at the previously processed element.
    # Example: if we are processing nums[j], q_prev = { OR(nums[i]...nums[j-1]) for all 0 <= i < j }.
    # The size of this set is at most B+1 (e.g., B=30 for 30-bit numbers, so size <= 31).
    q_prev = set()
    
    # Iterate through each number in the input list.
    # Each 'current_num' is nums[j] for some j, serving as the rightmost element of subarrays.
    for current_num in nums:
        # q_curr will store the set of distinct OR values of subarrays ending at 'current_num'.
        # Example: if nums=[a,b,c] and current_num is 'c', q_curr will store distinct values from {c, b|c, a|b|c}.
        q_curr = set()
        
        # Case 1: The subarray is just [current_num]. Its OR value is 'current_num'.
        q_curr.add(current_num)
        
        # Case 2: Subarrays are nums[i...j] where nums[j] is 'current_num' and i < j.
        # Such an OR value is OR(nums[i]...nums[j-1]) | current_num.
        # The term OR(nums[i]...nums[j-1]) is one of the values in q_prev.
        for prev_or_result in q_prev:
            q_curr.add(prev_or_result | current_num)
        
        # Now q_curr contains all distinct OR values for subarrays ending at 'current_num'.
        # Update min_abs_difference using these values.
        for val_in_q_curr in q_curr:
            min_abs_difference = min(min_abs_difference, abs(val_in_q_curr - k))
        
        # Optimization: if minimum difference is 0, it's the smallest possible absolute difference.
        if min_abs_difference == 0:
            return 0 # Return immediately
            
        # For the next number in 'nums', the set q_curr becomes the q_prev.
        q_prev = q_curr
            
    # All numbers processed. min_abs_difference holds the minimum.
    # It might be a float if initialized with math.inf (e.g. min(math.inf, 5) is 5.0).
    # The return type is int, so cast it.
    return int(min_abs_difference)