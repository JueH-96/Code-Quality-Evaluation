import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))

    # DP state: dp[last_char][state]
    # last_char: 0 or 1 (the value of T[k])
    # state:
    #   0: T[0...k] is alternating (no same adjacent pairs in T[0...k])
    #   1: T[0...k] has exactly one same pair, which is T[k-1] == T[k]
    #   2: T[0...k] has exactly one same pair, which is T[p] == T[p+1] for some p < k-1
    
    # Initialize dp table for k=0 (prefix T[0])
    # dp[last_char][state]
    # Use two rows for space optimization: dp_prev and dp_curr
    # dp_prev will store results for step k-1, dp_curr for step k
    dp_prev = [[float('inf')] * 3 for _ in range(2)]

    # k=0, prefix T[0]. A single character string is always considered alternating.
    # T[0] = 0
    cost_s0_to_0 = C[0] if S[0] == '1' else 0
    dp_prev[0][0] = cost_s0_to_0
    
    # T[0] = 1
    cost_s0_to_1 = C[0] if S[0] == '0' else 0
    dp_prev[1][0] = cost_s0_to_1

    # States 1 and 2 are impossible for a prefix of length 1.
    # dp_prev[0][1] = dp_prev[0][2] = dp_prev[1][1] = dp_prev[1][2] = float('inf') # Already initialized

    # Loop for k from 1 to N-1 (processing S[k] and determining T[k])
    for k in range(1, N):
        dp_curr = [[float('inf')] * 3 for _ in range(2)]
        
        cost_sk_to_0 = C[k] if S[k] == '1' else 0
        cost_sk_to_1 = C[k] if S[k] == '0' else 0

        # Calculate dp_curr[0][state] (Target T[k] = 0)
        
        # State 0 (T[0...k] alternating, ends 0): Requires T[k-1]=1 and T[0...k-1] alt (state 0).
        # T[k-1]=1, T[k]=0. This pair is different. If T[0...k-1] was alt, T[0...k] is also alt.
        if dp_prev[1][0] != float('inf'):
            dp_curr[0][0] = dp_prev[1][0] + cost_sk_to_0
        
        # State 1 (T[0...k] one same pair T[k-1]=T[k]=0): Requires T[k-1]=0 and T[0...k-1] alt (state 0).
        # T[k-1]=0, T[k]=0. This pair is same. If T[0...k-1] was alt, this is the first same pair.
        if dp_prev[0][0] != float('inf'):
            dp_curr[0][1] = dp_prev[0][0] + cost_sk_to_0
        
        # State 2 (T[0...k] one same pair p < k-1, ends 0): Requires T[k-1]=1.
        # T[k-1]=1, T[k]=0. This pair is different.
        # T[0...k-1] must have had exactly one same pair at p < k-1. This pair could be at k-2 (state 1 at k-1) or p < k-2 (state 2 at k-1).
        min_prev_state_for_0_2 = min(dp_prev[1][1], dp_prev[1][2])
        if min_prev_state_for_0_2 != float('inf'):
            dp_curr[0][2] = min_prev_state_for_0_2 + cost_sk_to_0

        # Calculate dp_curr[1][state] (Target T[k] = 1)

        # State 0 (T[0...k] alternating, ends 1): Requires T[k-1]=0 and T[0...k-1] alt (state 0).
        # T[k-1]=0, T[k]=1. This pair is different. If T[0...k-1] was alt, T[0...k] is also alt.
        if dp_prev[0][0] != float('inf'):
            dp_curr[1][0] = dp_prev[0][0] + cost_sk_to_1
        
        # State 1 (T[0...k] one same pair T[k-1]=T[k]=1): Requires T[k-1]=1 and T[0...k-1] alt (state 0).
        # T[k-1]=1, T[k]=1. This pair is same. If T[0...k-1] was alt, this is the first same pair.
        if dp_prev[1][0] != float('inf'):
            dp_curr[1][1] = dp_prev[1][0] + cost_sk_to_1

        # State 2 (T[0...k] one same pair p < k-1, ends 1): Requires T[k-1]=0.
        # T[k-1]=0, T[k]=1. This pair is different.
        # T[0...k-1] must have had exactly one same pair at p < k-1. This pair could be at k-2 (state 1 at k-1) or p < k-2 (state 2 at k-1).
        min_prev_state_for_1_2 = min(dp_prev[0][1], dp_prev[0][2])
        if min_prev_state_for_1_2 != float('inf'):
            dp_curr[1][2] = min_prev_state_for_1_2 + cost_sk_to_1
        
        # After calculating dp_curr for step k, it becomes dp_prev for step k+1
        dp_prev = dp_curr

    # The final good string T (length N, processed up to index N-1) must have exactly one same pair.
    # This corresponds to prefixes T[0...N-1] ending in state 1 or state 2.
    min_cost = float('inf')
    
    # Final string T[0...N-1] has exactly one same pair.
    # This single pair could be at index N-2 (state 1) or at index p < N-2 (state 2).
    
    # Final string ends with T[N-1]=0
    min_cost = min(min_cost, dp_prev[0][1]) # Single same pair at N-2
    min_cost = min(min_cost, dp_prev[0][2]) # Single same pair at p < N-2

    # Final string ends with T[N-1]=1
    min_cost = min(min_cost, dp_prev[1][1]) # Single same pair at N-2
    min_cost = min(min_cost, dp_prev[1][2]) # Single same pair at p < N-2

    print(min_cost)

solve()