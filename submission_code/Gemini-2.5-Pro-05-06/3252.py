from typing import List

class Solution:
  def incremovableSubarrayCount(self, nums: List[int]) -> int:
    n = len(nums)
    # Constraints: 1 <= nums.length <= 50. So n >= 1.

    # is_prefix_SI[k] is True if nums[0...k-1] (prefix of length k) is strictly increasing.
    # Array size is n+1, indices from 0 to n represent lengths.
    is_prefix_SI = [False] * (n + 1)
    is_prefix_SI[0] = True # Empty prefix (length 0) is strictly increasing.
    if n >= 1: # Always true by constraints
      is_prefix_SI[1] = True # Prefix of length 1 (e.g., nums[0:1]) is strictly increasing.
    
    for length in range(2, n + 1):
      # Prefix nums[0:length] (length 'length') is SI if:
      # 1. Prefix nums[0:length-1] (length 'length-1') is SI.
      # 2. The last two elements nums[length-2] and nums[length-1] are in strictly increasing order.
      if is_prefix_SI[length-1] and nums[length-2] < nums[length-1]:
        is_prefix_SI[length] = True
      # else: it remains False, as initialized.
            
    # is_suffix_SI[k] is True if nums[k...n-1] (suffix starting at index k) is strictly increasing.
    # Array size is n+1, indices from 0 to n represent start indices.
    # is_suffix_SI[k] refers to suffix nums[k:n].
    is_suffix_SI = [False] * (n + 1)
    is_suffix_SI[n] = True # Empty suffix (e.g., nums[n:n], starts at index n) is SI.
    if n >= 1: # Always true by constraints
      is_suffix_SI[n-1] = True # Suffix of length 1 (e.g., nums[n-1:n], starts at n-1) is SI.
    
    for start_idx in range(n - 2, -1, -1): # Iterate from n-2 down to 0
      # Suffix nums[start_idx:n] is SI if:
      # 1. Suffix nums[start_idx+1:n] is SI.
      # 2. The first two elements nums[start_idx] and nums[start_idx+1] are in strictly increasing order.
      if is_suffix_SI[start_idx+1] and nums[start_idx] < nums[start_idx+1]:
        is_suffix_SI[start_idx] = True
      # else: it remains False, as initialized.

    count = 0
    # Iterate over all possible subarrays to remove: nums[i_rem...j_rem]
    # i_rem: start index of removed subarray, from 0 to n-1
    # j_rem: end index of removed subarray, from i_rem to n-1
    for i_rem in range(n): 
      for j_rem in range(i_rem, n): 
        # Prefix remaining is nums[0...i_rem-1]. This is slice nums[0:i_rem]. Its length is i_rem.
        cond1_prefix_is_SI = is_prefix_SI[i_rem]
        
        # Suffix remaining is nums[j_rem+1...n-1]. This is slice nums[j_rem+1:n]. It starts at index j_rem+1.
        cond2_suffix_is_SI = is_suffix_SI[j_rem+1]
        
        cond3_connection_valid = True
        # Check connection only if both prefix and suffix are non-empty.
        # Prefix nums[0:i_rem] is non-empty if i_rem > 0.
        # Suffix nums[j_rem+1:n] is non-empty if j_rem+1 < n.
        
        prefix_non_empty = (i_rem > 0)
        # j_rem is at most n-1. So j_rem+1 is at most n.
        # Suffix is non-empty if its start index j_rem+1 is less than n.
        suffix_non_empty = (j_rem + 1 < n) 
        
        if prefix_non_empty and suffix_non_empty:
          # Last element of prefix is nums[i_rem-1].
          # First element of suffix is nums[j_rem+1].
          if nums[i_rem-1] >= nums[j_rem+1]:
            cond3_connection_valid = False
            
        if cond1_prefix_is_SI and cond2_suffix_is_SI and cond3_connection_valid:
          count += 1
          
    return count