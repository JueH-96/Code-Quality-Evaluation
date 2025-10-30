import math # math is not actually used, k.bit_length() is an int method. Could be removed.
from typing import List

class Solution:
  """
  Solves the problem using binary lifting (jump pointers) to efficiently calculate
  the sum of player IDs over k passes for each possible starting player.
  The function f(x) is defined as the sum of the starting player's ID and the IDs
  of all players who receive the ball during the k passes.
  f(x) = x + receiver[x] + receiver[receiver[x]] + ... + receiver^k[x].
  This can be written as P_0 + P_1 + ... + P_k, where P_0 = x and P_i = receiver(P_{i-1}).
  """
  def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
    """
    Finds the maximum function value f(x) over all possible starting players x.

    Args:
      receiver: A list where receiver[i] is the ID of the player who receives
                the ball from player i. The length of the list is N.
      k: The number of passes.

    Returns:
      The maximum value of f(x).
    """
    N = len(receiver)
    
    # Calculate P = number of levels needed for binary lifting.
    # P is the smallest integer such that 2^P > k.
    # Or equivalently, P = k.bit_length(). If k=4 (100_2), P=3. We need levels 0, 1, 2.
    # If k=1 (1_2), P=1. We need level 0.
    # The constraints state k >= 1, so P will be at least 1.
    P = k.bit_length() 
    
    # dp[i][p] stores a tuple: (final_node, path_sum)
    # final_node: The player ID after 2^p passes starting from player i. 
    #             This is equivalent to receiver^(2^p)[i]. Let's call it J(i, p).
    # path_sum: The sum of player IDs on the path P_0, P_1, ..., P_{2^p - 1}, where P_0 = i.
    #           This sum includes the starting node i. Let's call it S(i, p).
    # The DP table size is N x P.
    # Initialize with placeholder values.
    dp = [[(0, 0)] * P for _ in range(N)]

    # Base case p=0: This represents 2^0 = 1 pass.
    for i in range(N):
        # After 1 pass starting from i, the ball is at receiver[i]. This is final_node J(i, 0).
        # The path for this step, according to the definition of S(i, p), is just P_0 = i. 
        # The sum S(i, 0) is therefore just i.
        dp[i][0] = (receiver[i], i) 

    # Fill DP table using dynamic programming / binary lifting recurrence.
    # Calculate values for p = 1 up to P-1.
    for p in range(1, P):
        for i in range(N):
            # To compute the state after 2^p passes starting from i:
            # Consider the path as two segments of 2^(p-1) passes each.
            
            # First segment: 2^(p-1) passes starting from i.
            # node1 is the player after 2^(p-1) passes. J(i, p-1)
            # sum1 is the sum of IDs P_0...P_{2^(p-1)-1}. S(i, p-1)
            node1, sum1 = dp[i][p-1] 
            
            # Second segment: 2^(p-1) passes starting from node1.
            # node2 is the player after another 2^(p-1) passes. J(node1, p-1) = J(J(i, p-1), p-1) = J(i, p).
            # sum2 is the sum of IDs for the path starting at node1 for 2^(p-1) steps. S(node1, p-1).
            # Note: Constraints guarantee 0 <= receiver[i] <= N-1, so node1 is always a valid index.
            node2, sum2 = dp[node1][p-1]
            
            # Combine results for 2^p steps:
            # The final node after 2^p steps is node2 = J(i, p).
            # The total path sum for P_0 ... P_{2^p - 1} is sum1 + sum2 = S(i, p-1) + S(J(i, p-1), p-1) = S(i, p).
            dp[i][p] = (node2, sum1 + sum2)

    # Calculate the maximum function value f(x) over all starting players x
    max_f_val = 0
    for x in range(N): # Iterate through all possible starting players x
        current_node = x
        # This variable will accumulate the sum P_0 + P_1 + ... + P_{k-1} during the simulation
        current_path_sum = 0 
        
        # Simulate k passes using the precomputed DP table (binary lifting).
        # Decompose k into powers of 2. For each power 2^p present in k's binary representation,
        # take the corresponding jump using the precomputed DP table state.
        for p in range(P):
            # Check if the p-th bit of k is set (i.e., if 2^p is part of k's sum decomposition).
            if (k >> p) & 1: 
                # Get the precomputed state for a jump of 2^p steps starting from current_node
                # jump_node = J(current_node, p)
                # jump_sum = S(current_node, p) = sum of IDs from P_{current_idx} to P_{current_idx + 2^p - 1}
                jump_node, jump_sum = dp[current_node][p]
                
                # Add the sum contribution of this path segment to the total path sum
                current_path_sum += jump_sum
                
                # Update the current node to the node after this jump
                current_node = jump_node
        
        # After processing all bits of k, we have simulated exactly k passes.
        # The function value f(x) = P_0 + P_1 + ... + P_k.
        # The loop accumulates the sum corresponding to the path segments. Let's analyze the total sum.
        # The variable `current_path_sum` accumulates the sums S(node, p) for the jumps taken.
        # The definition of S(node, p) is sum P_{idx} ... P_{idx + 2^p - 1}.
        # This means `current_path_sum` correctly accumulates P_0 + ... + P_{k-1}.
        # The variable `current_node` holds the ID of the player P_k (the player holding the ball after k passes).
        # Therefore, f(x) = P_0 + ... + P_{k-1} + P_k = current_path_sum + current_node.
        f_x = current_path_sum + current_node
        
        # Update the maximum function value found so far
        max_f_val = max(max_f_val, f_x)
            
    # Return the overall maximum function value
    return max_f_val