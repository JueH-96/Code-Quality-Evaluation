# YOUR CODE HERE
import math
import sys

def solve():
    """
    Solves the race checkpoint problem using dynamic programming.
    Finds the minimum total cost (sum of Euclidean distances traveled + penalty) 
    to travel from checkpoint 1 to checkpoint N. Checkpoints 2 through N-1 can be skipped.
    The penalty depends exponentially on the number of skipped checkpoints C:
    - 0 if C = 0
    - 2^(C-1) if C > 0
    
    The dynamic programming state dp[c][i] represents the minimum distance traveled
    to reach checkpoint i (0-based index) having skipped exactly c checkpoints among 
    the intermediate checkpoints visited so far (checkpoints P_2 through P_i).
    The final answer minimizes dp[c][N-1] + Penalty(c) over possible values of c.
    """
    
    N = int(sys.stdin.readline())
    
    # Read coordinates for N checkpoints. Store them in a list of lists/tuples.
    coords = []
    for _ in range(N):
        coords.append(list(map(int, sys.stdin.readline().split())))

    # Handle the base case where N=2 separately for simplicity, although the main DP logic works.
    # If N=2, we must visit checkpoints 1 and N. No skips are possible (C=0).
    # The cost is just the distance between them.
    if N == 2: 
        dist_1_2 = math.dist(coords[0], coords[1])
        print(f"{dist_1_2:.17f}")
        return

    # C_limit: Maximum number of skips to consider in the DP state.
    # The penalty P(C) = 2^(C-1) grows very rapidly. If P(C) exceeds the maximum possible
    # travel distance, it's unlikely that skipping C checkpoints is optimal.
    # Maximum possible distance is roughly N * MaxCoord * sqrt(2). For N=10^4, MaxCoord=10^4,
    # this is ~1.4e8. 2^(C-1) > 1.4e8 implies C-1 >= 28, so C >= 29.
    # We use C_limit = 60 as a safe upper bound, significantly larger than 29.
    # The actual maximum possible number of skips is N-2 (when visiting only 1 and N).
    # If N-1 < 60, we only need to consider up to N-2 skips.
    # The DP table dimension for c needs to cover range [0, C_limit].
    # So we use min(60, N-1) because N-1 gives enough room for max N-2 skips (index N-2 needs size N-1).
    C_limit = min(60, N - 1) 

    # Initialize DP table `dp[c][i]`.
    # dp[c][i]: Minimum distance to reach checkpoint i (0-based index, corresponds to P_{i+1}) 
    # having skipped exactly c checkpoints among P_2, ..., P_i.
    # Dimensions: (C_limit + 1) rows for c=0..C_limit, N columns for i=0..N-1.
    # Initialize all distances to infinity.
    dp = [[float('inf')] * N for _ in range(C_limit + 1)]

    # Base case: Reaching checkpoint 0 (P_1) requires 0 distance and involves 0 skips.
    dp[0][0] = 0.0 

    # Fill the DP table using iterative approach.
    # Iterate through target checkpoints P_{i+1} (index i) from 1 to N-1.
    for i in range(1, N): 
        # Iterate through possible number of skips c to reach checkpoint i.
        for c in range(C_limit + 1): 
            
            # Optimization: Determine the minimum index `p` for the previous checkpoint.
            # The number of skips to reach p, c_prev, must be non-negative.
            # c = c_prev + skips_step = c_prev + (i - p - 1)
            # c_prev = c - (i - p - 1) >= 0  =>  p >= i - c - 1.
            # This optimization significantly reduces the number of `p` values to check for large `c`.
            min_p_idx = max(0, i - c - 1) 
            
            # Iterate through possible previous checkpoints P_{p+1} (index p) from min_p_idx up to i-1.
            for p in range(min_p_idx, i): 
                
                # Calculate the number of checkpoints skipped when moving directly from P_{p+1} to P_{i+1}.
                # These are checkpoints P_{p+2}, ..., P_i. Their count is max(0, i - (p+1)) = max(0, i - p - 1).
                skips_step = i - p - 1 
                
                # Calculate the number of skips required to reach the previous checkpoint p.
                c_prev = c - skips_step 
                
                # Check if the state for the previous checkpoint (p, c_prev) is valid and reachable.
                # c_prev must be within the valid range [0, C_limit].
                if 0 <= c_prev <= C_limit: 
                    # Check if the previous state dp[c_prev][p] is reachable (not infinity).
                    if dp[c_prev][p] != float('inf'): 
                        
                        # Calculate Euclidean distance between checkpoints p and i.
                        dist_p_i = math.dist(coords[p], coords[i])
                        
                        # Calculate the total distance to reach checkpoint i via p with c total skips.
                        current_dist = dp[c_prev][p] + dist_p_i
                        
                        # Update dp[c][i] if this path offers a shorter distance.
                        dp[c][i] = min(dp[c][i], current_dist)

    # After filling the DP table, find the overall minimum total cost to reach the final checkpoint N.
    min_total_cost = float('inf')

    # Iterate through all possible skip counts c (from 0 to C_limit) for reaching the final checkpoint N (index N-1).
    for c in range(C_limit + 1):
        # Check if the final checkpoint is reachable with c skips.
        if dp[c][N-1] != float('inf'):
            # Calculate the penalty based on the number of skips c.
            penalty = 0.0
            if c > 0:
                # Penalty is 2^(c-1). Use integer exponentiation (left shift) for precision 
                # with potentially large powers (up to 2^59), then convert to float.
                # This handles c=1 correctly: 1 << (1-1) = 1 << 0 = 1.
                penalty = float(1 << (c - 1)) 
                
            # Calculate total cost: distance sum to reach N-1 + penalty.
            total_cost = dp[c][N-1] + penalty
            
            # Update the overall minimum total cost found so far.
            min_total_cost = min(min_total_cost, total_cost)

    # Print the minimum total cost found, formatted to the required precision.
    print(f"{min_total_cost:.17f}")

# Execute the solver function
solve()