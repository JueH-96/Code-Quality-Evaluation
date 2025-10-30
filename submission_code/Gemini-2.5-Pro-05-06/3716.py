from typing import List

class Solution:
  def longestSubsequence(self, nums: List[int]) -> int:
    n = len(nums)
    # Constraints: 2 <= nums.length.
    # Min length is 1 (single element). Max diff is 299 (300-1). Max diff index is 299.
    # dp[i][d]: length of longest valid subsequence ending at nums[i] with last diff d.
    #           Initialized to 0. Must be >= 2 to have a difference.
    dp = [[0] * 300 for _ in range(n)]

    # smax_arrays[i][d]: max(dp[i][k] for k >= d). Suffix maximums for dp[i] row.
    smax_arrays = [[0] * 300 for _ in range(n)]

    overall_max_len = 1 # Min subsequence length is 1.

    # For i = 0, dp[0] and smax_arrays[0] remain all 0s as no predecessor exists.
    # overall_max_len = 1 accounts for [nums[0]].

    for i in range(n):
      # max_val_in_dp_i_row is the max length of a subsequence (>=2 elements) ending at nums[i].
      # Used to update overall_max_len.
      max_val_in_dp_i_row = 0
      
      for j in range(i): # nums[j] is the potential predecessor of nums[i]
        diff = abs(nums[i] - nums[j])

        # Case 1: Subsequence [nums[j], nums[i]]. Length is 2.
        # dp[i][diff] is initially 0. Max with 2.
        if 2 > dp[i][diff]: # This check is essentially dp[i][diff] = max(dp[i][diff], 2)
             dp[i][diff] = 2
        
        # Case 2: Extend subsequence [..., prev_val, nums[j]] with nums[i].
        # Need prev_diff >= diff. Max length from dp[j][prev_diff] is smax_arrays[j][diff].
        val_from_smax_j = smax_arrays[j][diff]
        if val_from_smax_j > 0: 
          # val_from_smax_j > 0 means a valid subsequence of length >= 2 ending at nums[j]
          # exists, satisfying the non-increasing difference condition for extension.
          extended_length = 1 + val_from_smax_j
          if extended_length > dp[i][diff]:
            dp[i][diff] = extended_length
      
      # After computing dp[i][...] for current i, compute smax_arrays[i][...].
      # And update overall_max_len using the max value found in dp[i][...].
      current_smax_val_for_row_i = 0
      for d_idx in range(299, -1, -1): # Iterate d_idx from 299 down to 0
        if dp[i][d_idx] > current_smax_val_for_row_i:
          current_smax_val_for_row_i = dp[i][d_idx]
        smax_arrays[i][d_idx] = current_smax_val_for_row_i
      
      # current_smax_val_for_row_i is now max(dp[i][...]).
      # This is the max length of a valid subsequence (>=2 elements) ending at nums[i].
      if current_smax_val_for_row_i > overall_max_len:
        overall_max_len = current_smax_val_for_row_i
            
    return overall_max_len