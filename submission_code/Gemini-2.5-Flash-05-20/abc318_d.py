# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())

    # Adjacency matrix to store edge weights.
    # Vertices are 0-indexed (0 to N-1) for convenience.
    # Initialize with 0s.
    adj = [[0] * N for _ in range(N)]

    # Read the edge weights from input.
    # The input format provides D_{i,j} where i < j.
    # We convert 1-based input indices (i, j) to 0-based indices (i-1, j-1).
    for i_0_indexed in range(N - 1): # Corresponds to original vertex i+1 (from 1 to N-1)
        # Read the line of weights for edges starting from (i_0_indexed + 1)
        current_line_weights = list(map(int, sys.stdin.readline().split()))
        
        weight_idx = 0
        # Iterate for the second vertex (j_0_indexed) which is always greater than i_0_indexed.
        # Corresponds to original vertex j+1 (from i+2 to N)
        for j_0_indexed in range(i_0_indexed + 1, N):
            adj[i_0_indexed][j_0_indexed] = current_line_weights[weight_idx]
            weight_idx += 1

    # Memoization dictionary to store results of subproblems.
    # `memo[mask]` will store the maximum weight of a matching
    # using only vertices present in the set represented by `mask`.
    memo = {}

    def max_matching_weight(mask):
        # If the result for this mask has already been computed, return it.
        if mask in memo:
            return memo[mask]

        # Find the smallest available vertex (lowest set bit) in the current mask.
        # This vertex `u` will be processed.
        # `(mask & -mask)` isolates the lowest set bit.
        # `.bit_length() - 1` converts the bit value to its 0-indexed position.
        # If mask is 0, (mask & -mask) is 0, and 0.bit_length() is 0, so u will be -1.
        u = (mask & -mask).bit_length() - 1
        
        # Base case: if no vertices are left in the mask (mask is 0),
        # no more edges can be chosen, so the accumulated weight is 0.
        if u == -1: # This condition implies mask == 0
            return 0

        # Option 1: Vertex 'u' is NOT part of any chosen edge in this matching.
        # We simply remove 'u' from the mask and recurse on the remaining vertices.
        # `mask ^ (1 << u)` unsets the bit corresponding to 'u'.
        res = max_matching_weight(mask ^ (1 << u))

        # Option 2: Vertex 'u' IS part of an edge (u, v).
        # We iterate through all other available vertices 'v' (where v > u to avoid duplicate edges and self-loops).
        # For each such 'v', we consider choosing the edge (u, v).
        for v in range(u + 1, N):
            if (mask >> v) & 1: # Check if 'v' is also available (its bit is set in mask).
                # If we choose edge (u, v):
                # 1. Add its weight `adj[u][v]`.
                # 2. Recurse on the subproblem where both 'u' and 'v' are removed from the mask.
                #    `mask ^ (1 << u) ^ (1 << v)` unsets bits for 'u' and 'v'.
                current_pair_weight = adj[u][v] + max_matching_weight(mask ^ (1 << u) ^ (1 << v))
                # Update the maximum result found so far.
                res = max(res, current_pair_weight)
        
        # Store the computed result in the memoization table before returning.
        memo[mask] = res
        return res

    # The initial call starts with all vertices available.
    # A mask with all N bits set is `(1 << N) - 1`.
    result = max_matching_weight((1 << N) - 1)
    print(result)

# Call the solve function to run the program.
solve()