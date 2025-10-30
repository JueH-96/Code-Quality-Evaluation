import math

class Solution:
  def minLength(self, s: str, numOps: int) -> int:
    n = len(s)
    s_numeric = [int(c) for c in s] # Convert '0'/'1' string to list of ints 0/1
    
    infinity = float('inf')

    # Memoization for can_achieve(k) results. Not strictly necessary for performance
    # in a standard binary search on k, but good practice if k could repeat.
    memo_can_achieve = {}

    def can_achieve(k: int) -> bool:
        # k is the target maximum length of identical char substrings
        
        # Base case: max length cannot be 0 for non-empty string (n>=1)
        if k == 0: 
            return False
        
        if k in memo_can_achieve:
            return memo_can_achieve[k]

        # prev_dp[char_val][length] stores min_flips for prefix s[0...i-1]
        # ending in char_val with run `length`.
        # Dimensions: 2 (for char 0/1) x (k+1) (for lengths 0 up to k).
        # Active lengths are 1 to k.
        prev_dp = [[infinity] * (k + 1) for _ in range(2)]
        
        # prev_min_ops_ending_char[char_val] stores min_flips to end prefix s[0...i-1]
        # with char_val (any valid run length up to k).
        prev_min_ops_ending_char = [infinity] * 2

        # --- Base case for index i = 0 (character s_numeric[0]) ---
        # Cost to make s_numeric[0] be 0
        cost_to_be_0_at_idx0 = 0 if s_numeric[0] == 0 else 1
        if cost_to_be_0_at_idx0 <= numOps: # Pruning: only consider paths not exceeding numOps
            prev_dp[0][1] = cost_to_be_0_at_idx0
            prev_min_ops_ending_char[0] = cost_to_be_0_at_idx0
        
        # Cost to make s_numeric[0] be 1
        cost_to_be_1_at_idx0 = 0 if s_numeric[0] == 1 else 1
        if cost_to_be_1_at_idx0 <= numOps:
            prev_dp[1][1] = cost_to_be_1_at_idx0
            prev_min_ops_ending_char[1] = cost_to_be_1_at_idx0

        # If processing s[0] already makes it impossible with numOps budget
        if prev_min_ops_ending_char[0] == infinity and prev_min_ops_ending_char[1] == infinity:
            memo_can_achieve[k] = False
            return False

        # --- Loop for indices i from 1 to n-1 (for character s_numeric[i]) ---
        for i in range(1, n):
            curr_dp = [[infinity] * (k + 1) for _ in range(2)]
            curr_min_ops_ending_char = [infinity] * 2 # Min ops for prefix s[0...i]
            
            for curr_char_val in range(2): # Target character for s_numeric[i] (0 or 1)
                cost_flip_current_char = 1 if s_numeric[i] != curr_char_val else 0
                
                # Case 1: Run length is 1 for s_numeric[i] (ends a run of the other char)
                # Prefix s[0...i-1] must have ended in (1 - curr_char_val).
                ops_prev_ended_diff_char = prev_min_ops_ending_char[1 - curr_char_val]
                if ops_prev_ended_diff_char != infinity:
                    new_ops = ops_prev_ended_diff_char + cost_flip_current_char
                    if new_ops <= numOps: # Pruning optimization
                       curr_dp[curr_char_val][1] = new_ops
                
                # Case 2: Run length `length_val` > 1 for s_numeric[i] (continues run of curr_char_val)
                # Prefix s[0...i-1] must have ended in `curr_char_val` with run `length_val-1`.
                for length_val in range(2, k + 1):
                    ops_prev_ended_same_char_len_minus_1 = prev_dp[curr_char_val][length_val - 1]
                    if ops_prev_ended_same_char_len_minus_1 != infinity:
                        new_ops = ops_prev_ended_same_char_len_minus_1 + cost_flip_current_char
                        if new_ops <= numOps: # Pruning optimization
                            curr_dp[curr_char_val][length_val] = new_ops
            
            # Update curr_min_ops_ending_char for current index i based on curr_dp
            for char_val_loop in range(2):
                min_for_this_char = infinity
                for length_val_loop in range(1, k + 1): # Iterate through all possible run lengths
                     min_for_this_char = min(min_for_this_char, curr_dp[char_val_loop][length_val_loop])
                curr_min_ops_ending_char[char_val_loop] = min_for_this_char
            
            # Current DP state becomes previous for the next iteration
            prev_dp = curr_dp 
            prev_min_ops_ending_char = curr_min_ops_ending_char

            # Early exit: if all ways to form prefix s[0...i] exceed numOps budget
            if min(prev_min_ops_ending_char[0], prev_min_ops_ending_char[1]) == infinity:
                memo_can_achieve[k] = False
                return False
        
        # After processing all n characters, final_min_ops is min ops for s[0...n-1]
        final_min_ops = min(prev_min_ops_ending_char[0], prev_min_ops_ending_char[1])
        
        result = (final_min_ops <= numOps) # Check if achievable within budget
        memo_can_achieve[k] = result
        return result

    # --- Binary search for the minimum k (target_max_len) ---
    low = 1         # Smallest possible max length
    high = n        # Largest possible max length
    ans = n         # Initialize ans with a safe upper bound

    while low <= high:
        mid = low + (high - low) // 2 # Candidate for max_len
        if can_achieve(mid):
            ans = mid            # mid is achievable, try for an even smaller max_len
            high = mid - 1
        else:
            low = mid + 1        # mid is not achievable, need a larger max_len
    
    return ans