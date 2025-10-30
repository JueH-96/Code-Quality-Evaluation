# YOUR CODE HERE
import sys

def fast_pow(base, power):
    """
    Calculates base^power % MOD efficiently using modular exponentiation.
    """
    result = 1
    MOD = 998244353
    # Reduce base initially if it's larger than MOD
    base %= MOD 
    while power > 0:
        # If power is odd, multiply result with base
        if power % 2 == 1:
            result = (result * base) % MOD
        # Square the base and halve the power
        base = (base * base) % MOD
        power //= 2
    return result

def solve():
    N = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    MOD = 998244353

    # Count the number of '1's in the string s. This is K.
    K = s.count('1')

    if K == 0:
        # Case 1: No vertices on the cycle are connected to vertex N.
        # Vertex N is isolated. The graph consists of the cycle C_N and vertex N.
        # The number of edges is |E| = N.
        # The total number of orientations is 2^N.
        # The in-degree of vertex N is always 0.
        # We need to count distinct sequences (d_0, ..., d_{N-1}) for the cycle C_N.
        # For a cycle C_N, there are 2^N orientations. The sequence (1, 1, ..., 1)
        # is achieved by two orientations (all clockwise or all counter-clockwise).
        # All other 2^N - 2 orientations yield distinct sequences.
        # The total number of distinct sequences is (2^N - 2)/1 + 1 = 2^N - 1.
        # The final sequence includes d_N=0.
        
        # Calculate (2^N - 1) mod MOD
        ans = (fast_pow(2, N) - 1 + MOD) % MOD
        print(ans)
    else:
        # Case 2: K >= 1. At least one vertex on the cycle is connected to vertex N.
        # The total number of edges is |E| = N + K.
        # The total number of orientations is 2^(N+K).
        
        # We analyze the number of distinct in-degree sequences (d_0, ..., d_N).
        # Let x = (x_0, ..., x_{N-1}) represent orientations of cycle edges, 
        # where x_i=1 means i -> (i+1) mod N, x_i=0 means (i+1) mod N -> i.
        # Let y = (y_i)_{i | s_i=1} represent orientations of edges connecting to N,
        # where y_i=1 means i -> N, y_i=0 means N -> i.
        
        # For a fixed choice of y, the in-degree of N is fixed: d_N = sum(y_i).
        # The in-degrees for vertices 0..N-1 are:
        # d_i = (1-x_i) + x_{i-1} + (1-y_i if s_i=1 else 0)
        
        # For a fixed y, the map from x to (d_0, ..., d_{N-1}) results in 2^N-1 distinct sequences.
        # This is because x=(0,...,0) and x=(1,...,1) map to the same sequence, 
        # while all other 2^N-2 choices of x map to distinct sequences.
        
        # Consider two different choices of y vectors, y and y'.
        # If sum(y_i) != sum(y'_i), the resulting sequences are distinct because d_N differs.
        # If sum(y_i) == sum(y'_i), the sets of sequences D(y) and D(y') might overlap.
        
        # It can be shown that the only pairs of orientations that result in the same 
        # in-degree sequence are pairs of the form:
        # Orientation 1: ((0, ..., 0), y)  (cycle edges all counter-clockwise based on index)
        # Orientation 2: ((1, ..., 1), y)  (cycle edges all clockwise based on index)
        # for any fixed choice of y.
        # There are 2^K possible choices for y. So there are 2^K such pairs.
        # Each pair corresponds to one distinct in-degree sequence that is obtained twice.
        
        # Total number of orientations = 2^(N+K).
        # Number of sequences obtained twice = 2^K.
        # Number of orientations contributing to these sequences = 2 * 2^K.
        # Number of sequences obtained once = (Total Orientations - Orientations for twice-obtained sequences) / 1
        #                                   = (2^(N+K) - 2 * 2^K) / 1
        #                                   = 2^(N+K) - 2^(K+1).
        # Total number of distinct sequences = (Sequences obtained twice) + (Sequences obtained once)
        #                                  = 2^K + (2^(N+K) - 2^(K+1))
        #                                  = 2^(N+K) - 2^K.
        
        # This formula also matches the K=1 case: 2^(N+1) - 2^1 = 2 * 2^N - 2 = 2(2^N - 1).

        # Calculate (2^(N+K) - 2^K) mod MOD
        term1 = fast_pow(2, N + K)
        term2 = fast_pow(2, K)
        
        ans = (term1 - term2 + MOD) % MOD
        print(ans)

solve()