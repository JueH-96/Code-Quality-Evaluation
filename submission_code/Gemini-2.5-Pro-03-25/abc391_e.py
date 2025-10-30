# YOUR CODE HERE
import sys

def solve():
    # Read the integer N from standard input
    N = int(sys.stdin.readline())
    # Read the binary string A from standard input and remove leading/trailing whitespace
    A = sys.stdin.readline().strip()
    
    # Calculate the length of the initial string A. The length is 3^N.
    L = 3**N

    # Initialize the dynamic programming table `dp_curr`. 
    # `dp_curr[k]` will store a tuple `(min_cost_to_get_0, min_cost_to_get_1)` 
    # representing the minimum number of changes required in the substring corresponding to index k
    # at the current level to make the resulting bit 0 or 1, respectively.
    # The base case is level 0, which corresponds to the individual characters of the input string A.
    dp_curr = []
    for i in range(L):
        if A[i] == '0':
            # If the character A[i] is '0':
            # The cost to get 0 is 0 (no change needed).
            # The cost to get 1 is 1 (one flip from 0 to 1 is needed).
            dp_curr.append((0, 1)) 
        else: # A[i] == '1'
            # If the character A[i] is '1':
            # The cost to get 0 is 1 (one flip from 1 to 0 is needed).
            # The cost to get 1 is 0 (no change needed).
            dp_curr.append((1, 0))

    # `current_len` keeps track of the number of elements in the `dp_curr` list, 
    # which corresponds to the length of the binary string at the current level of operations.
    current_len = L
    
    # We need to apply the majority operation N times. This loop simulates these N operations.
    # Each iteration calculates the DP values for the next level based on the current level's values.
    for _ in range(N): 
        # Calculate the length of the sequence for the next level. It's reduced by a factor of 3.
        next_len = current_len // 3
        # Initialize `dp_next` list to store the DP results for the next level.
        # Fill it with placeholder tuples (0, 0). The size is `next_len`.
        dp_next = [(0, 0)] * next_len 
        
        # Iterate through each group k in the next level. The number of groups is `next_len`.
        for k in range(next_len):
            # Determine the indices of the three elements/groups in the current level (`dp_curr`) 
            # that form the k-th group for the next level. These indices are 3k, 3k+1, 3k+2.
            idx1 = 3*k
            idx2 = 3*k + 1
            idx3 = 3*k + 2

            # Retrieve the minimum costs from the `dp_curr` table for these three "children" nodes.
            # `c0x` is the minimum cost to make child x result in 0.
            # `c1x` is the minimum cost to make child x result in 1.
            c01, c11 = dp_curr[idx1]
            c02, c12 = dp_curr[idx2]
            c03, c13 = dp_curr[idx3]

            # Calculate the minimum cost to achieve a majority value of 0 for the current group k.
            # The majority is 0 if the three children's values are (0,0,0), (0,0,1), (0,1,0), or (1,0,0).
            # For each configuration, sum the minimum costs to achieve those child values.
            cost0_options = [
                c01 + c02 + c03, # Cost required if children are (0,0,0)
                c01 + c02 + c13, # Cost required if children are (0,0,1)
                c01 + c12 + c03, # Cost required if children are (0,1,0)
                c11 + c02 + c03  # Cost required if children are (1,0,0)
            ]
            # The overall minimum cost to get 0 is the minimum of the costs for these four configurations.
            min_cost0 = min(cost0_options)

            # Calculate the minimum cost to achieve a majority value of 1 for the current group k.
            # The majority is 1 if the three children's values are (1,1,1), (1,1,0), (1,0,1), or (0,1,1).
            # Sum the minimum costs for each required child value configuration.
            cost1_options = [
                c11 + c12 + c13, # Cost required if children are (1,1,1)
                c11 + c12 + c03, # Cost required if children are (1,1,0)
                c11 + c02 + c13, # Cost required if children are (1,0,1)
                c01 + c12 + c13  # Cost required if children are (0,1,1)
            ]
            # The overall minimum cost to get 1 is the minimum of the costs for these four configurations.
            min_cost1 = min(cost1_options)

            # Store the computed minimum costs (cost for 0, cost for 1) for group k in the `dp_next` table.
            dp_next[k] = (min_cost0, min_cost1)
        
        # After processing all groups for the current level, update `dp_curr` to `dp_next`
        # to prepare for the next iteration (next level of operation).
        dp_curr = dp_next
        # Update the length of the sequence representation.
        current_len = next_len

    # After N iterations, `dp_curr` contains exactly one element: the tuple (final_cost0, final_cost1)
    # representing the minimum costs to make the final single bit result 0 or 1, starting from the initial string A.
    final_cost0, final_cost1 = dp_curr[0]

    # The dynamic programming computes the minimum cost starting from the original string A.
    # This means the cost associated with the *original* final value (without any changes) must be 0.
    # If the original final value was 0, then `final_cost0` will be 0. The problem asks for the minimum cost
    # to *change* this value, which means achieving 1. This cost is `final_cost1`.
    # If the original final value was 1, then `final_cost1` will be 0. The minimum cost to change this value
    # means achieving 0. This cost is `final_cost0`.
    if final_cost0 == 0:
        # Original final value was 0. The answer is the cost to flip it to 1.
        print(final_cost1)
    else: # This implies final_cost1 == 0
        # Original final value was 1. The answer is the cost to flip it to 0.
        print(final_cost0)

# Standard Python entry point check: ensures `solve()` is called only when the script is executed directly.
if __name__ == '__main__':
    solve()
# YOUR CODE ENDS HERE