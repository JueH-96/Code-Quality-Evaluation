import collections
from typing import List

class Solution:
  def maximumLength(self, nums: List[int], k: int) -> int:
    n = len(nums)
    # According to constraints, n >= 1.
    
    # dp[val][changes] = max_length of a good subsequence ending with val, 
    #                    using 'changes' number of changes.
    # dp is a dictionary: {value_from_nums: {num_changes: length}}
    # collections.defaultdict(lambda: collections.defaultdict(int)) ensures that if a key 
    # is accessed and not found, it defaults to 0.
    dp = collections.defaultdict(lambda: collections.defaultdict(int))

    # max_len_for_changes[c] stores the maximum length of a good subsequence
    # using exactly 'c' changes, ending with ANY value encountered so far (from nums[0...i-1]).
    max_len_for_changes = [0] * (k + 1)
    
    ans = 0 # Stores the maximum length found across all states.

    # Iterate through each number in the input array
    for i in range(n):
      num = nums[i]
      
      # Iterate number of changes 'c' from k down to 0.
      # This order is crucial: when calculating for (num, c), we need values of 
      # dp[num][c] and max_len_for_changes[c-1] that are based on nums[0...i-1].
      # Iterating 'c' downwards ensures that max_len_for_changes[c-1] (used for current 'c')
      # has not yet been updated by the current 'num' in this outer loop pass.
      # Similarly, dp[num][c_val] for different c_val are updated independently in this pass.
      for c in range(k, -1, -1):
        # Path 1: Extend a sequence that already ends with `num` and used `c` changes.
        # The length of such a sequence (based on nums[0...i-1]) is dp[num][c].
        # Appending current `num` increases length by 1, changes count remains `c`.
        len_extending_same_val = dp[num][c] 
        if len_extending_same_val > 0: # Must be a valid existing sequence
          len_extending_same_val += 1
        # If dp[num][c] was 0, len_extending_same_val remains 0 (path not possible).
        
        # Path 2: Extend a sequence that ends with `X` (X != num) and used `c-1` changes.
        # The max length of such a sequence (based on nums[0...i-1]) is max_len_for_changes[c-1].
        # Appending current `num` increases length by 1, changes count becomes `c`.
        # This path is only possible if c > 0.
        len_extending_diff_val = 0
        if c > 0:
          max_len_prev_c_minus_1 = max_len_for_changes[c-1]
          if max_len_prev_c_minus_1 > 0: # Must be a valid existing sequence
            len_extending_diff_val = max_len_prev_c_minus_1 + 1
        # If max_len_for_changes[c-1] was 0, len_extending_diff_val remains 0.
        
        # Determine the new maximum length for a subsequence ending with `num` using `c` changes.
        new_max_len_for_num_c = 0
        if c == 0:
          # For 0 changes:
          # Option A: The subsequence is just `[num]`. Length 1.
          # Option B: Extend a 0-change sequence `...num` to `...num, num`. (covered by len_extending_same_val).
          new_max_len_for_num_c = max(1, len_extending_same_val)
        else: # c > 0
          # Choose the better of Path 1 or Path 2.
          # If both are 0, it means this state (ending with `num`, using `c` changes) is not reachable yet.
          new_max_len_for_num_c = max(len_extending_same_val, len_extending_diff_val)

        # If a valid length (>0) is found for this state (current num, c changes):
        if new_max_len_for_num_c > 0:
          # Update the max length for a subsequence ending with `num` using `c` changes.
          # This value will be used in the next iteration of the outer loop (for nums[i+1]).
          dp[num][c] = new_max_len_for_num_c
          
          # Update the overall maximum length for any subsequence using `c` changes.
          # This value will be used for calculations involving nums[i+1].
          max_len_for_changes[c] = max(max_len_for_changes[c], new_max_len_for_num_c)
          
          # Update the overall answer.
          ans = max(ans, new_max_len_for_num_c)
            
    return ans