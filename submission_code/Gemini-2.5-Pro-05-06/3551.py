class Solution:
  def maximumSubarrayXor(self, nums: list[int], queries: list[list[int]]) -> list[int]:
    n = len(nums)

    # xor_scores[p_val][s_idx] stores the XOR score of nums[s_idx...s_idx+p_val]
    # p_val is length-1, actual length is p_val+1
    # p_val ranges from 0 to n-1 (effectively serves as row index)
    # s_idx ranges from 0 to n-1-p_val (effectively serves as column index)
    
    # Using N*N tables for simplicity; loops will ensure only valid parts are used.
    xor_scores = [[0] * n for _ in range(n)]

    # Base case: p_val=0 (subarrays of length 1)
    for s_idx in range(n):
      xor_scores[0][s_idx] = nums[s_idx]

    # Fill xor_scores table using the recurrence
    for p_val in range(1, n): 
      for s_idx in range(n - p_val): 
        xor_scores[p_val][s_idx] = xor_scores[p_val-1][s_idx] ^ xor_scores[p_val-1][s_idx+1]

    # MaxScoreStartingAt[s_idx][k_val] stores max_{p_iter=0...k_val} xor_scores[p_iter][s_idx]
    # s_idx ranges from 0 to n-1 (row index for this table)
    # k_val (which is a p_val) ranges from 0 to n-1-s_idx (column index for this table)
    
    MaxScoreStartingAt = [[0] * n for _ in range(n)] 

    for s_idx in range(n): # start_index of subarray
      MaxScoreStartingAt[s_idx][0] = xor_scores[0][s_idx] 
      # k_val is p_val, a length-1 specifier. Max k_val for this s_idx is (n-1-s_idx).
      for k_val in range(1, n - s_idx): 
        MaxScoreStartingAt[s_idx][k_val] = max(MaxScoreStartingAt[s_idx][k_val-1], xor_scores[k_val][s_idx])
    
    # Process queries
    ans_list = []
    for L, R in queries:
      max_overall = 0
      # s_val is the start index of a potential subarray
      for s_val in range(L, R + 1):
        # p_val_for_query is the length-1 for a subarray starting at s_val and ending at or before R.
        # Max length-1 for subarray nums[s_val...s_val+p_val_for_query] where s_val+p_val_for_query <= R
        # is p_val_for_query = R - s_val.
        max_p_val_for_this_s = R - s_val
        
        # MaxScoreStartingAt[s_val][p] gives the max score of any subarray starting at s_val
        # with length-1 up to p. This is exactly what we need for the fixed s_val.
        # Indices are valid: s_val is in [0, n-1].
        # max_p_val_for_this_s is in [0, n-1-s_val] because L <= s_val <= R and R <= n-1.
        max_overall = max(max_overall, MaxScoreStartingAt[s_val][max_p_val_for_this_s])
      ans_list.append(max_overall)
      
    return ans_list