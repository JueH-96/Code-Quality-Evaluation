# YOUR CODE HERE
import sys

# Set higher recursion depth limit if using recursion; not strictly needed for iterative DP
# sys.setrecursionlimit(2000) 

def solve():
    # Read N, the number of vertices, from Standard Input
    N = int(sys.stdin.readline())
    
    # Handle edge cases N=0 and N=1. 
    # Problem constraints state N >= 2, but defensive programming is good.
    # For N=0 or N=1, no edges can be formed, so the maximum weight is 0.
    if N <= 1:
        print(0)
        return

    # Read the edge weights according to the specified input format.
    # The input gives weights D_{i,j} for vertices i and j (1-based indexing) where 1 <= i < j <= N.
    # The weights are provided row by row. The k-th row (0-based index in the list) 
    # corresponds to edges starting from vertex k+1 (1-based index).
    weights_list = []
    # There will be N-1 lines of input weights following N.
    for _ in range(N - 1):
        weights_list.append(list(map(int, sys.stdin.readline().split())))

    # Initialize an N x N adjacency matrix `W` to store edge weights.
    # We will use 0-based indexing for vertices internally (0 to N-1) for easier bitmask manipulation.
    # W[i][j] will store the weight of the edge between vertex i+1 and vertex j+1 (using 1-based names from problem).
    W = [[0] * N for _ in range(N)]
    
    # Populate the weight matrix `W` using the read weights.
    current_list_idx = 0 # Tracks which row of weights_list we are currently processing
    # Iterate through vertices i from 0 to N-2 (representing original vertices 1 to N-1)
    for i in range(N - 1): 
        row_weights = weights_list[current_list_idx]
        k = 0 # Index within the current row_weights list
        # Iterate through vertices j from i+1 to N-1 (representing original vertices i+2 to N)
        for j in range(i + 1, N): 
            weight = row_weights[k]
            # Store the weight for the edge between 0-based indices i and j
            W[i][j] = weight
            # Since the graph is undirected, the weight matrix is symmetric. W[j][i] = W[i][j].
            W[j][i] = weight 
            k += 1
        current_list_idx += 1

    # Initialize the DP table. dp[mask] will store the maximum weight matching
    # achievable using only the vertices present in the subset represented by `mask`.
    # A mask is an integer where the k-th bit is set (1) if vertex k (0-based index) is in the subset, and 0 otherwise.
    # The size of the DP table is 2^N. Initialize all entries to 0.
    dp = [0] * (1 << N)

    # Iterate through all possible non-empty subsets of vertices.
    # Subsets are represented by bitmasks from 1 up to (2^N - 1).
    # The order of iteration ensures that for any mask, dp values for all its submasks are computed beforehand.
    for mask in range(1, 1 << N):
        
        # Find any vertex `p` belonging to the current subset `mask`.
        # A common and efficient strategy is to choose the vertex corresponding to the least significant bit set in `mask`.
        p = 0
        # This loop finds the index `p` such that the p-th bit is the lowest set bit in mask.
        # This means vertex `p` (0-based index) is in the subset represented by `mask`.
        while not (mask & (1 << p)):
            p += 1

        # Calculate dp[mask] using the dynamic programming recurrence relation.
        # Consider two possibilities for the chosen vertex `p` in an optimal matching for the subset `mask`:
        
        # Case 1: Vertex `p` is NOT matched with any other vertex within the subset `mask`.
        # In this situation, the maximum weight matching for subset `mask` is the same as the
        # maximum weight matching for the subset `mask` excluding `p`.
        # The mask for this smaller subset is `mask ^ (1 << p)`.
        # The value `dp[mask ^ (1 << p)]` has already been computed because `mask ^ (1 << p)` is numerically smaller than `mask`.
        option1_val = dp[mask ^ (1 << p)]
        
        # Case 2: Vertex `p` IS matched with some other vertex `q` within the subset `mask`.
        # We need to find the best partner `q` for `p` that maximizes the total weight.
        option2_max_val = 0 # Variable to keep track of the maximum weight found for this case.
        
        # Iterate through all potential partners `q` for vertex `p`.
        # `q` must also be in the subset `mask`.
        # To avoid double counting pairs (since edge (p,q) is same as (q,p)) and self-loops (p=q), 
        # we only need to check vertices `q` such that `q > p`.
        for q in range(p + 1, N):
            # Check if vertex `q` (0-based index) is present in the current subset `mask`.
            # This is done by checking if the q-th bit is set in `mask`.
            if (mask & (1 << q)):
                # If `p` is matched with `q`:
                # The total weight for this choice is the weight of edge (p, q) plus the maximum weight matching
                # that can be formed using the remaining vertices in the subset (i.e., `mask` without `p` and `q`).
                # The mask for the remaining subset is `mask ^ (1 << p) ^ (1 << q)`.
                # The DP value `dp[remaining_mask]` for this smaller subset has already been computed.
                remaining_mask = mask ^ (1 << p) ^ (1 << q)
                current_val = W[p][q] + dp[remaining_mask]
                
                # Update the maximum weight found for Case 2 if `current_val` is greater than the current maximum.
                option2_max_val = max(option2_max_val, current_val)
        
        # The value of dp[mask] is the maximum achievable weight for the subset `mask`.
        # This is the maximum of the weight obtained in Case 1 (p unmatched) and Case 2 (p matched with best q).
        dp[mask] = max(option1_val, option2_max_val)

    # The final answer is the maximum weight matching using all N vertices {1, ..., N}.
    # In our 0-based indexing, this corresponds to the subset {0, ..., N-1}.
    # The mask representing this full set has all N bits set, which is (1 << N) - 1.
    print(dp[(1 << N) - 1])

# Call the solve function to execute the main logic of the program
solve()