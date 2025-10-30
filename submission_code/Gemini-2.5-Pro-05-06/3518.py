import math # Not strictly needed for float('-inf') but good practice if other math constants/functions were used.

class Solution:
  def maxScore(self, a: list[int], b: list[int]) -> int:
    N = len(b)
    
    # dp_prev stores results from previous iteration (for a_idx-1)
    # dp_curr stores results for current iteration (for a_idx)
    dp_prev = [float('-inf')] * N 
    dp_curr = [float('-inf')] * N

    # Base case: using a[0] (this corresponds to a hypothetical a_idx = 0)
    # The (a_idx+1)-th element chosen is the 1st element.
    # j_b is the index in array b. For the 1st element, j_b must be at least 0.
    for j_b in range(N): # Loop for j_b from 0 to N-1
        dp_curr[j_b] = a[0] * b[j_b]
    
    # Iteratively fill for a_idx = 1, 2, 3 
    # (These are indices in array `a`. They correspond to choosing the 2nd, 3rd, and 4th element from `b`)
    for a_idx in range(1, 4): # a_idx takes values 1, 2, 3
        # Results from the previous iteration (for a_idx-1) are in dp_curr.
        # These become the "previous" results for the current iteration.
        # So, dp_prev takes values from dp_curr.
        # dp_curr will be reset and filled for the current a_idx.
        dp_prev, dp_curr = dp_curr, dp_prev 
        
        # Reset dp_curr for the current iteration.
        # (It now holds values from two iterations ago, or the initial float('-inf') if a_idx=1)
        for i in range(N):
             dp_curr[i] = float('-inf')

        max_val_from_prev_iter_dp_states = float('-inf')
        
        # For a[a_idx], we are picking b[j_b] as the (a_idx+1)-th element chosen from b.
        # The index j_b must be at least a_idx.
        # Example: if a_idx=1 (for a[1]), we are choosing the 2nd element from b.
        # This 2nd element b[j_b] must have j_b >= 1, because the 1st element b[p] has p < j_b and p >= 0.
        # So, smallest p is 0, thus smallest j_b for the 2nd element is 1.
        for j_b in range(a_idx, N):
            # max_val_from_prev_iter_dp_states is the maximum of dp_prev[p] 
            # for p from (a_idx-1) up to (j_b-1).
            # The value dp_prev[j_b-1] represents the max score for choosing `a_idx` elements from b,
            # with the (a_idx)-th element being b[j_b-1].
            # The index for this (a_idx)-th element (which is j_b-1) must be at least (a_idx-1).
            # This condition (j_b-1 >= a_idx-1) is satisfied because our loop for j_b starts from a_idx.
            
            if dp_prev[j_b-1] != float('-inf'): # Ensure dp_prev[j_b-1] is a valid (reachable) state
                max_val_from_prev_iter_dp_states = max(max_val_from_prev_iter_dp_states, dp_prev[j_b-1])
            
            # If max_val_from_prev_iter_dp_states is still float('-inf'), it means no valid prefix sequence was found.
            if max_val_from_prev_iter_dp_states != float('-inf'):
                dp_curr[j_b] = max_val_from_prev_iter_dp_states + a[a_idx] * b[j_b]
    
    # After the loops, dp_curr contains results for a_idx=3 (i.e., 4 elements chosen, using a[0]..a[3]).
    # The final answer is the maximum value in dp_curr.
    # The 4th element chosen from b (b[j_b]) must be at an index j_b >= 3.
    final_max_score = float('-inf')
    for j in range(3, N): # Smallest possible index for the 4th chosen element is 3.
        final_max_score = max(final_max_score, dp_curr[j])
            
    return int(final_max_score)