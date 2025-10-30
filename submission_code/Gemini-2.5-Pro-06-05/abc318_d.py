# YOUR CODE HERE
import sys

def solve():
    """
    Solves the maximum weight matching problem on a complete graph
    using dynamic programming with bitmasking.
    """
    
    # Read the number of vertices, N.
    try:
        line = sys.stdin.readline()
        if not line.strip():
            return
        N = int(line)
    except (IOError, ValueError):
        return

    # Store weights in a symmetric N x N matrix D.
    # Vertices are 0-indexed internally (0 to N-1).
    D = [[0] * N for _ in range(N)]
    if N > 1:
        weights_flat = []
        # Input format is the upper triangle of the weight matrix.
        for _ in range(N - 1):
            weights_flat.extend(map(int, sys.stdin.readline().split()))
        
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                D[i][j] = D[j][i] = weights_flat[k]
                k += 1

    # dp[mask] will store the maximum weight of a perfect matching on the
    # vertices represented by the bitmask 'mask'.
    dp = [0] * (1 << N)

    # Iterate through all subsets of vertices (masks).
    # The order of iteration ensures that when we compute dp[mask],
    # the values for all its submasks are already computed.
    for mask in range(1, 1 << N):
        
        # A perfect matching requires an even number of vertices.
        # If the number of vertices in the subset (popcount) is odd, skip.
        if bin(mask).count('1') % 2 != 0:
            continue
            
        # Find the first vertex 'p' in the current subset.
        # This vertex must be paired with some other vertex 'q' in the subset.
        p = 0
        while not (mask & (1 << p)):
            p += 1

        # To compute dp[mask], we try pairing 'p' with every other vertex 'q'
        # in the subset and take the one that maximizes the total weight.
        max_for_this_mask = 0
        for q in range(p + 1, N):
            if mask & (1 << q):
                # If we pair (p, q), the remaining vertices form a subproblem.
                prev_mask = mask ^ (1 << p) ^ (1 << q)
                
                # The total weight for this choice is D[p][q] plus the max weight
                # perfect matching on the remaining vertices (dp[prev_mask]).
                current_total_weight = D[p][q] + dp[prev_mask]
                
                if current_total_weight > max_for_this_mask:
                    max_for_this_mask = current_total_weight
        
        dp[mask] = max_for_this_mask

    # The problem asks for the maximum weight matching. Any such matching is a
    # perfect matching on some subset of the vertices.
    # Our dp table contains the maximum weight perfect matching for every subset.
    # Therefore, the answer is the maximum value in the entire dp table.
    print(max(dp))

solve()