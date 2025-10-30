import bisect

class Solution:
  def maximumWeight(self, intervals: list[list[int]]) -> list[int]:
    n = len(intervals)
    if n == 0:
      return [] # Should not happen based on constraints (1 <= intervals.length)
      
    # Augment intervals with original indices: (l, r, w, original_idx)
    s_intervals = []
    for i in range(n):
      s_intervals.append((intervals[i][0], intervals[i][1], intervals[i][2], i))
    
    # Sort intervals. Python's default tuple sort is element by element.
    # So, this sorts by l_i, then r_i, then w_i, then original_idx_i.
    # Primarily sorting by start points (l_i) is key.
    s_intervals.sort()
    
    # Precompute start points for efficient binary search
    start_points = [s_intervals[i][0] for i in range(n)]
    
    K_max = 4 # Maximum number of intervals to choose
    
    # dp[i][k] stores a pair: (max_score, list_of_original_indices)
    # This represents the best choice considering intervals from s_intervals[i:] 
    # (i.e., s_intervals[i], s_intervals[i+1], ..., s_intervals[n-1])
    # and selecting at most k non-overlapping intervals.
    #
    # Initialize with (0, []) representing zero score and empty list of indices.
    # This is the correct base case if no intervals are chosen or can be chosen.
    # dp table dimensions: (n+1) rows for indices i=0..n
    #                      (K_max+1) columns for k=0..K_max
    dp = [[(0, []) for _ in range(K_max + 1)] for _ in range(n + 1)]
    
    # Iterate i from n-1 down to 0
    for i in range(n - 1, -1, -1):
      l_i, r_i, w_i, original_idx_i = s_intervals[i]
      
      # Find the smallest index j > i such that s_intervals[j].l > r_i.
      # This means s_intervals[j].l must be at least r_i + 1.
      # bisect_left(array, value, lo, hi) searches in array[lo...hi-1].
      # We search in start_points corresponding to s_intervals[i+1 ... n-1].
      # If no such interval exists, next_j will be n.
      next_j = bisect.bisect_left(start_points, r_i + 1, lo=i + 1, hi=n)
      
      for k in range(1, K_max + 1):
        # Option 1: Skip s_intervals[i]
        # The result is dp[i+1][k] (max score from S[i+1:] with at most k choices).
        res_skip = dp[i+1][k]
        
        # Option 2: Pick s_intervals[i]
        # We use one interval (s_intervals[i]), so we need at most k-1 more.
        # These k-1 intervals must be chosen from s_intervals[next_j:].
        # The best result for this is dp[next_j][k-1].
        score_from_remaining, indices_from_remaining = dp[next_j][k-1]
        
        # Construct the list of indices for res_pick.
        # It must be sorted for lexicographical comparison.
        # indices_from_remaining is already sorted (by DP invariant).
        # Create a new list; do not modify indices_from_remaining from dp table.
        current_indices_list_for_pick = sorted([original_idx_i] + indices_from_remaining)
        res_pick = (w_i + score_from_remaining, current_indices_list_for_pick)
        
        # Compare res_pick and res_skip to update dp[i][k]
        if res_pick[0] > res_skip[0]:
          dp[i][k] = res_pick
        elif res_skip[0] > res_pick[0]:
          dp[i][k] = res_skip
        else: # Scores are equal, tie-break with lexicographically smallest index list
          # Python's list comparison is inherently lexicographical and handles
          # empty lists and lists of different lengths correctly.
          if res_pick[1] < res_skip[1]:
            dp[i][k] = res_pick
          else: # res_skip[1] <= res_pick[1] (if lists are equal, pick_skip is fine)
            dp[i][k] = res_skip
            
    # The final answer is for s_intervals[0:] (all intervals) using at most K_max items.
    # We need to return only the list of indices.
    return dp[0][K_max][1]