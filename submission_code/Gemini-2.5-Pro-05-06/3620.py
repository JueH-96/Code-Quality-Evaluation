import math
from typing import List

class Solution:
  def maxDistinctElements(self, nums: List[int], k: int) -> int:
    nums.sort()
    
    distinct_count = 0
    # Initialize last_selected_val to a value that is guaranteed to be smaller
    # than any potential first chosen value.
    # The first chosen value will be at least nums[0] - k.
    # nums[i] >= 1, k >= 0. Smallest possible nums[i] - k is 1 - 10^9.
    # float('-inf') is the simplest way to represent a value smaller than this.
    # Note: -math.inf + 1 results in -math.inf, which is handled correctly by max().
    last_selected_val = -math.inf 
    
    for num in nums:
        # The current number `num` can be transformed into any integer
        # in the range [num - k, num + k].
        min_reachable_for_current_num = num - k
        max_reachable_for_current_num = num + k
        
        # We want to pick a target value for the current `num`.
        # To maximize the number of distinct elements in our greedy construction,
        # this target value must be strictly greater than `last_selected_val`.
        # The smallest integer strictly greater than `last_selected_val` is `last_selected_val + 1`.
        desired_next_val = last_selected_val + 1
        
        # However, the value chosen for `num` cannot be smaller than `min_reachable_for_current_num`.
        # So, the actual target value we attempt to pick (candidate) must be at least `min_reachable_for_current_num`.
        # Therefore, the smallest possible value we can pick for `num` (that is > last_selected_val) is:
        candidate_val = max(min_reachable_for_current_num, desired_next_val)
        
        # Now, we check if this `candidate_val` is achievable,
        # i.e., if it's less than or equal to `max_reachable_for_current_num`.
        if candidate_val <= max_reachable_for_current_num:
            # If yes, we "select" this `candidate_val` for the current `num`.
            # This becomes the new `last_selected_val` for the next iteration.
            last_selected_val = candidate_val
            distinct_count += 1
            # If `candidate_val > max_reachable_for_current_num`, it means we cannot pick a 
            # distinct value for the current `num` that is greater than `last_selected_val`
            # while also respecting `num`'s own range `[num-k, num+k]`.
            # In this case, `distinct_count` is not incremented, and `last_selected_val`
            # remains unchanged (it represents the last successfully picked distinct value).
            # We then move to process the next number in `nums`.
            
    return distinct_count