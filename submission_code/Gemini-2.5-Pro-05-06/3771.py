import collections

class Solution:
  def maxSubstringLength(self, s: str, k: int) -> bool:
    n = len(s)
    
    if k == 0:
      return True
    
    # Constraints: n >= 2. If k > 0 and n < 2, it's false.
    # But n >= 2 implies s cannot be empty or single char.
    
    # Precompute first and last occurrences of each character
    first = [-1] * 26  # Stores first index of char 'a'+i
    last = [-1] * 26   # Stores last index of char 'a'+i
    
    for i in range(n):
      char_val = ord(s[i]) - ord('a')
      if first[char_val] == -1:
        first[char_val] = i
      last[char_val] = i
      
    # dp[p] = max number of disjoint special substrings in s[0...p-1]
    # dp array is of size n + 1. dp[0] = 0.
    # dp[j+1] corresponds to prefix s[0...j]
    dp = [0] * (n + 1) 
    
    for j_end in range(n): # j_end is the 0-indexed end of substring s[i_start...j_end]
      # Option 1: Don't end any special substring at j_end.
      # dp[j_end+1] (for prefix s[0...j_end]) takes value from dp[j_end] (for prefix s[0...j_end-1]).
      # Note: dp elements are initialized to 0. Max with current value is fine.
      dp[j_end+1] = max(dp[j_end+1], dp[j_end])

      # Variables to track character span for the current window s[i_start...j_end]
      # Max last occurrence index among characters in s[i_start...j_end]
      current_window_max_last = -1 
      # Min first occurrence index among characters in s[i_start...j_end]
      current_window_min_first = n   
      
      # Option 2: Try to end a special substring s[i_start...j_end] at j_end.
      # Iterate i_start from j_end down to 0.
      for i_start in range(j_end, -1, -1):
        char_val_at_i_start = ord(s[i_start]) - ord('a')
        
        # Update span based on s[i_start]
        current_window_max_last = max(current_window_max_last, last[char_val_at_i_start])
        current_window_min_first = min(current_window_min_first, first[char_val_at_i_start])
        
        # Condition check: If any character in s[i_start...j_end] must appear after j_end,
        # then s[i_start...j_end] cannot be special.
        # Also, any s[x...j_end] (where x < i_start) would also include this character,
        # so it also cannot be special. Thus, break from the inner loop (over i_start).
        if current_window_max_last > j_end:
          break
        
        # If all checks pass so far, it means for all characters c in s[i_start...j_end]:
        #   last[c] <= j_end (guaranteed by the break condition above)
        #   And current_window_max_last holds the maximum of these last[c] values.
        # We also need first[c] >= i_start for all c. This is captured if current_window_min_first == i_start.
        
        # Check for "special" and "tight" conditions:
        if current_window_min_first == i_start and current_window_max_last == j_end:
          # s[i_start...j_end] is a candidate.
          # Check "not entire string" condition.
          if not (i_start == 0 and j_end == n - 1):
            # It's a valid special substring.
            # num_prev_substrings = dp[i_start] (max substrings in s[0...i_start-1])
            # If i_start is 0, dp[0] = 0, meaning 0 previous substrings.
            num_prev_substrings = dp[i_start] 
            dp[j_end+1] = max(dp[j_end+1], num_prev_substrings + 1)
            
    return dp[n] >= k