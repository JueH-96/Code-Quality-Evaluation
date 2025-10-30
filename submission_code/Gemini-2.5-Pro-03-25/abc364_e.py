# YOUR CODE HERE
import sys

def solve():
    # Read input values for N, X, Y
    N, X, Y = map(int, sys.stdin.readline().split())
    
    # Read dish properties (sweetness A_i, saltiness B_i)
    dishes = []
    for _ in range(N):
        dishes.append(list(map(int, sys.stdin.readline().split())))

    # Initialize DP table
    # dp[i][j] stores the minimum total saltiness achieved using exactly 'i' dishes
    # such that the total sweetness is exactly 'j'.
    # Dimensions: (N+1) rows for number of dishes (0 to N)
    #             (X+1) columns for total sweetness (0 to X)
    # Initialize all entries to infinity, representing unreachable states.
    dp = [[float('inf')] * (X + 1) for _ in range(N + 1)]
    
    # Base case: 0 dishes result in 0 sweetness and 0 saltiness.
    # This state is reachable and valid.
    dp[0][0] = 0
    
    # Variable to track the maximum number of dishes Snuke can eat.
    # Initialize to 0, as eating 0 dishes is always possible.
    max_k_found = 0

    # Iterate through each dish to potentially include it in a sequence
    for k in range(N): # Process dish k (0-indexed)
        Ak, Bk = dishes[k] # Sweetness and Saltiness of the current dish k
        
        # Iterate 'i' downwards. 'i' represents the number of dishes used
        # *before* considering adding the current dish 'k'.
        # The maximum number of dishes used before considering dish k is k.
        # Thus, the loop for 'i' goes from k down to 0.
        # Iterating downwards is crucial for 0/1 knapsack type problems
        # to ensure that we use values from the DP state *before* processing dish k.
        for i in range(k, -1, -1): 
            
            # Iterate through all possible total sweetness values 'j_prev' (from 0 to X)
            # achieved using 'i' dishes before considering dish k.
            for j_prev in range(X + 1): 
                
                # Check if the state (i dishes, sweetness j_prev) is reachable (dp value is not infinity)
                if dp[i][j_prev] != float('inf'):
                    
                    # Check the condition based on problem rules:
                    # The state *before* eating dish k must be within the limits.
                    # Total sweetness j_prev <= X is guaranteed by the loop range.
                    # Total saltiness dp[i][j_prev] must be <= Y.
                    if dp[i][j_prev] <= Y:
                        
                        # If the state (i, j_prev) is valid (within limits), 
                        # Snuke can potentially eat dish k.
                        # Eating dish k results in a sequence of length i+1.
                        # Update the maximum sequence length found so far.
                        max_k_found = max(max_k_found, i + 1)
                        
                        # Calculate the new state after eating dish k
                        new_sweetness = j_prev + Ak
                        new_saltiness = dp[i][j_prev] + Bk
                        
                        # Check if the new total sweetness is within the bounds of the DP table (<= X).
                        if new_sweetness <= X:
                             # If yes, update the DP table for the state (i+1 dishes, new_sweetness).
                             # We store the minimum saltiness found so far to reach this state.
                             # This state potentially allows extending the sequence even further.
                             # Note: We update dp[i+1][new_sweetness] even if new_saltiness > Y.
                             # This is because the check for stopping occurs *after* eating the dish.
                             # If new_saltiness > Y, this state (i+1, new_sweetness) won't be used
                             # to extend the path further because the condition `dp[i+1][new_sweetness] <= Y`
                             # will fail in the next iteration step.
                             dp[i + 1][new_sweetness] = min(dp[i + 1][new_sweetness], new_saltiness)

    # After iterating through all dishes and updating the DP table,
    # max_k_found holds the maximum number of dishes Snuke can eat according to the rules.
    # Print the final result.
    print(max_k_found)

# Call the solve function to execute the program logic
solve()