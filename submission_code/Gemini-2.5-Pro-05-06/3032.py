from typing import List

class Solution:
  def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
    n = len(receiver)
    
    # The function f(x) is the sum of k+1 terms:
    # x_0, x_1, ..., x_k
    # where x_0 = x, and x_j = receiver[x_{j-1}] for j > 0.
    # This sum has k+1 terms.
    num_terms = k + 1
    
    # Determine max_p: number of levels for DP table.
    # max_p will be num_terms.bit_length().
    # e.g., if num_terms = 5 (binary 101), bit_length is 3. Levels p=0,1,2 needed.
    # Constraints: k >= 1, so num_terms >= 2. Thus, num_terms.bit_length() >= 2.
    max_p = num_terms.bit_length()

    # dp_next[p][i]: player ID after 2^p passes, starting from player i.
    # dp_sum[p][i]: sum of IDs (i + receiver^1[i] + ... + receiver^(2^p - 1)[i]).
    # This sum consists of 2^p player IDs from the path.
    dp_next = [[0] * n for _ in range(max_p)]
    dp_sum = [[0] * n for _ in range(max_p)]
    
    # Base case: p = 0 (corresponds to 2^0 = 1 term in sum, and 2^0 = 1 pass to find next player)
    # After 2^0 = 1 pass starting from i, the player is receiver[i].
    # The sum of IDs for a path segment of length 2^0=1 starting at i is just i itself.
    for i in range(n):
        dp_next[0][i] = receiver[i]
        dp_sum[0][i] = i 

    # Fill DP tables for p from 1 up to max_p - 1
    # Each p corresponds to paths covering 2^p terms.
    for p in range(1, max_p):
        for i in range(n):
            # Player reached after 2^(p-1) passes starting from i. 
            # This is the start of the second half of the 2^p path segment.
            intermediate_player = dp_next[p-1][i]
            
            # Player after 2^p passes = (2^(p-1) passes from i) then (another 2^(p-1) passes from intermediate_player)
            dp_next[p][i] = dp_next[p-1][intermediate_player]
            
            # Sum for a path corresponding to 2^p terms:
            # (sum for first 2^(p-1) terms starting at i) + (sum for next 2^(p-1) terms starting at intermediate_player)
            dp_sum[p][i] = dp_sum[p-1][i] + dp_sum[p-1][intermediate_player]
            
    max_overall_sum = 0
    
    # Calculate f(x) for each starting player x (represented by index i here)
    for i in range(n): 
        current_sum_for_this_start_node = 0
        current_player_in_path = i # This is x_0, the starting player
        
        # Decompose num_terms into powers of 2 (using its binary representation)
        # and sum up corresponding dp_sum segments.
        # For example, if num_terms = 5 (binary 101), we need sum for a 2^0 segment and a 2^2 segment.
        # Iterate through each bit position of num_terms.
        for bit_idx in range(max_p): # bit_idx corresponds to p in dp_tables, i.e., power 2^bit_idx
            if (num_terms >> bit_idx) & 1: # Check if the (bit_idx)-th bit of num_terms is set
                # Add sum of the segment of length 2^bit_idx starting at current_player_in_path
                current_sum_for_this_start_node += dp_sum[bit_idx][current_player_in_path]
                # Advance current_player_in_path by 2^bit_idx steps to find the start of the next segment
                current_player_in_path = dp_next[bit_idx][current_player_in_path]
        
        if current_sum_for_this_start_node > max_overall_sum:
            max_overall_sum = current_sum_for_this_start_node
            
    return max_overall_sum