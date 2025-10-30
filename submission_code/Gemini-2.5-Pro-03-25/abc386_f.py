# YOUR CODE HERE
import sys

# Function to read input faster
input = sys.stdin.readline

def solve():
    # Read K, the maximum allowed number of operations
    K = int(input())
    # Read string S
    S = input().strip()
    # Read string T
    T = input().strip()
    
    # Get lengths of strings S and T
    N = len(S)
    M = len(T)

    # Optimization: Calculate the difference in lengths between S and T.
    # If this difference is greater than K, it's impossible to make them identical
    # within K operations. This is because each operation (insert, delete, substitute)
    # changes the length difference by at most 1 (insert/delete changes by 1, substitute by 0).
    # So, at least |N - M| insert/delete operations are needed.
    if abs(N - M) > K:
        print("No")
        return

    # Use dynamic programming with space optimization and banded computation.
    # dp_prev stores the DP values for the previous row (row i-1).
    # It's implemented as a dictionary where keys are column indices j,
    # and values are the minimum edit distance dp[i-1][j].
    # We only store states (cells) where the edit distance is <= K.
    dp_prev = {}
    
    # Initialize DP table for row 0.
    # dp[0][j] represents the cost of transforming an empty string "" (prefix of S of length 0)
    # into the prefix T[0...j-1]. This requires j insertion operations.
    # So, dp[0][j] = j.
    # We only need to initialize states for j where dp[0][j] <= K, because states with
    # cost > K are not useful. The maximum relevant j is min(M, K).
    for j in range(min(M + 1, K + 1)):
         dp_prev[j] = j # The cost is j insertions.

    # Iterate through rows i from 1 to N. Each row corresponds to processing the prefix S[0...i-1].
    for i in range(1, N + 1):
        # dp_curr will store DP values for the current row i. Initialize as empty dictionary.
        dp_curr = {}
        
        # Determine the relevant range of columns [j_min, j_max] for the current row i.
        # We use the property that if dp[i][j] <= k, then |i - j| <= k.
        # This limits the computation to a band around the main diagonal.
        # The band is defined by i-K <= j <= i+K.
        # We must also keep j within the valid bounds [0, M].
        j_min = max(0, i - K)
        j_max = min(M, i + K)

        # dp_i_jm1 stores the value of dp[i][j-1], which is the state immediately to the left
        # in the current row. This value is needed for the insertion cost calculation for dp[i][j].
        # Initialize to K + 1, representing a cost greater than K (effectively infinity for our purposes).
        dp_i_jm1 = K + 1 

        # Handle the base case for the current row: dp[i][0].
        # This represents transforming the prefix S[0...i-1] to an empty string "".
        # This requires i deletion operations, so dp[i][0] = i.
        # This state is only relevant if j=0 falls within the processing band (i.e., j_min == 0).
        # We only consider this state if its cost i is within the budget K.
        if j_min == 0: # Check if column 0 is relevant for this row
            if i <= K: # Check if the cost is within budget
                dp_curr[0] = i # Store the state and cost
                # Update dp_i_jm1. This value dp[i][0] will be used as dp[i][j-1]
                # when calculating the cost for dp[i][1] in the next step of the inner loop.
                dp_i_jm1 = i 
            # If i > K, then dp[i][0] > K. We don't store this state.
            # dp_i_jm1 correctly remains K+1 initially, indicating high cost for insertion originating from dp[i][0].

        # Iterate through columns j in the relevant band [j_min, j_max] for the current row i.
        for j in range(j_min, j_max + 1):
            
            # Skip the computation for j=0 here because it was already handled by the base case logic above.
            if j == 0:
                continue 

            # Compute dp[i][j] using the standard Levenshtein distance recurrence relation:
            # dp[i][j] = min(cost_deletion, cost_insertion, cost_substitution_or_match)
            
            # 1. Deletion cost: Transition from state dp[i-1][j]. Delete character S[i-1].
            # Cost is dp[i-1][j] + 1.
            # Fetch dp[i-1][j] from dp_prev dictionary. If key j is not found, it means dp[i-1][j] > K.
            # Use K+1 as default value to represent a cost > K.
            cost_del = dp_prev.get(j, K + 1) + 1
            
            # 2. Insertion cost: Transition from state dp[i][j-1]. Insert character T[j-1].
            # Cost is dp[i][j-1] + 1.
            # The value dp[i][j-1] is stored in dp_i_jm1 from the previous iteration of this inner loop.
            cost_ins = dp_i_jm1 + 1
            
            # 3. Substitution/Match cost: Transition from state dp[i-1][j-1]. 
            # If S[i-1] == T[j-1], it's a match, cost is dp[i-1][j-1].
            # If S[i-1] != T[j-1], it's a substitution, cost is dp[i-1][j-1] + 1.
            # Fetch dp[i-1][j-1] from dp_prev. Use K+1 default if key j-1 not found.
            cost_sub = dp_prev.get(j - 1, K + 1)
            if S[i-1] != T[j-1]: # Check if characters match
                cost_sub += 1 # Add 1 for substitution if they don't match
            
            # Find the minimum cost among the three possible operations.
            min_cost = min(cost_del, cost_ins, cost_sub)
            
            # Check if the minimum cost is within the allowed budget K.
            if min_cost <= K:
                # If yes, store this state and its minimum cost in the DP table for the current row.
                dp_curr[j] = min_cost
                # Update dp_i_jm1 to this minimum cost. It will be used as dp[i][j] (the left neighbor)
                # when computing the cost for dp[i][j+1] in the next iteration of the loop.
                dp_i_jm1 = min_cost 
            else:
                # If minimum cost exceeds K, this state is too expensive to reach within the budget.
                # We do not store it in dp_curr (so the key `j` will be absent).
                # Update dp_i_jm1 to K+1 to signify infinite cost. This ensures that
                # any path passing through this expensive state will also correctly reflect a cost > K.
                dp_i_jm1 = K + 1 

        # After computing all relevant states for row i, update dp_prev to dp_curr.
        # The dp_curr dictionary now becomes the 'previous' row's DP state for the next iteration (i+1).
        dp_prev = dp_curr
        
        # Optimization: Check if the dp_prev dictionary is empty. If it is, it means
        # no state currently reachable within the computation band has a cost <= K.
        # This implies that the target state dp[N][M] cannot possibly be reached within K operations.
        # We can terminate the algorithm early.
        if not dp_prev:
             print("No")
             return

    # After iterating through all N rows (processing all prefixes of S)
    # The final answer depends on the value of dp[N][M].
    # Check if the target column index M exists as a key in the final dp_prev map.
    # Its existence implies dp[N][M] <= K because we only store states with cost <= K.
    if M in dp_prev: 
      print("Yes") # Possible within K operations
    else:
      # If M is not in dp_prev, it means dp[N][M] > K. This could be because the computed cost
      # was > K, or M was outside the computation band for row N (less likely given initial check).
      # In either case, it's impossible within K operations.
      print("No")

# Call the solve function to execute the main logic of the program
solve()