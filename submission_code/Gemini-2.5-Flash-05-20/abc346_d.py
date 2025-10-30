import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    C = list(map(int, sys.stdin.readline().split()))

    # Helper function to get cost of transforming S[idx] to target_val (int 0 or 1)
    def get_cost(idx, target_val):
        return C[idx] if int(S[idx]) != target_val else 0

    # Precompute prefix sums for cost to match pattern A (0101...) and pattern B (1010...)
    # pref_A_cost[i] stores cost to make S[0...i] match 0101... pattern
    # pref_B_cost[i] stores cost to make S[0...i] match 1010... pattern
    pref_A_cost = [0] * N
    pref_B_cost = [0] * N

    # Initialize first elements (index 0)
    pref_A_cost[0] = get_cost(0, 0) # Pattern A expects 0 at index 0
    pref_B_cost[0] = get_cost(0, 1) # Pattern B expects 1 at index 0

    for i in range(1, N):
        # Target for pattern A at index i: 0 if i is even, 1 if i is odd
        target_A_char_at_i = i % 2 
        pref_A_cost[i] = pref_A_cost[i-1] + get_cost(i, target_A_char_at_i)

        # Target for pattern B at index i: 1 if i is even, 0 if i is odd
        target_B_char_at_i = 1 - (i % 2)
        pref_B_cost[i] = pref_B_cost[i-1] + get_cost(i, target_B_char_at_i)

    # Precompute suffix sums for cost to match pattern A and pattern B
    # suff_A_cost[i] stores cost to make S[i...N-1] match 0101... pattern
    # suff_B_cost[i] stores cost to make S[i...N-1] match 1010... pattern
    suff_A_cost = [0] * N
    suff_B_cost = [0] * N

    # Initialize last elements (index N-1)
    suff_A_cost[N-1] = get_cost(N-1, (N-1) % 2)
    suff_B_cost[N-1] = get_cost(N-1, 1 - ((N-1) % 2))

    for i in range(N - 2, -1, -1):
        # Target for pattern A at index i: 0 if i is even, 1 if i is odd
        target_A_char_at_i = i % 2
        suff_A_cost[i] = suff_A_cost[i+1] + get_cost(i, target_A_char_at_i)

        # Target for pattern B at index i: 1 if i is even, 0 if i is odd
        target_B_char_at_i = 1 - (i % 2)
        suff_B_cost[i] = suff_B_cost[i+1] + get_cost(i, target_B_char_at_i)
    
    min_total_cost = float('inf')

    # Iterate through all possible break points k (0-indexed, where T[k] == T[k+1])
    # k ranges from 0 to N-2
    for k in range(N - 1):
        # Scenario 1: T[k] = 0 and T[k+1] = 0
        current_cost_00 = get_cost(k, 0) + get_cost(k+1, 0)

        # Cost for prefix S[0...k-1]:
        # T[k] needs to be 0. If k is even, pattern A (0101...) sequence ends with 0 at k.
        # If k is odd, pattern B (1010...) sequence ends with 0 at k.
        if k > 0: # If prefix exists
            if k % 2 == 0: # T[k]=0 matches pattern A at even k
                current_cost_00 += pref_A_cost[k-1]
            else: # T[k]=0 matches pattern B at odd k
                current_cost_00 += pref_B_cost[k-1]
        
        # Cost for suffix S[k+2...N-1]:
        # T[k+1] needs to be 0. If (k+1) is even, pattern A sequence starts with 0 at k+1.
        # If (k+1) is odd, pattern B sequence starts with 0 at k+1.
        if k + 2 < N: # If suffix exists
            if (k + 1) % 2 == 0: # T[k+1]=0 matches pattern A at even k+1
                current_cost_00 += suff_A_cost[k+2]
            else: # T[k+1]=0 matches pattern B at odd k+1
                current_cost_00 += suff_B_cost[k+2]
        
        min_total_cost = min(min_total_cost, current_cost_00)

        # Scenario 2: T[k] = 1 and T[k+1] = 1
        current_cost_11 = get_cost(k, 1) + get_cost(k+1, 1)

        # Cost for prefix S[0...k-1]:
        # T[k] needs to be 1. If k is even, pattern B (1010...) sequence ends with 1 at k.
        # If k is odd, pattern A (0101...) sequence ends with 1 at k.
        if k > 0: # If prefix exists
            if k % 2 == 0: # T[k]=1 matches pattern B at even k
                current_cost_11 += pref_B_cost[k-1]
            else: # T[k]=1 matches pattern A at odd k
                current_cost_11 += pref_A_cost[k-1]
        
        # Cost for suffix S[k+2...N-1]:
        # T[k+1] needs to be 1. If (k+1) is even, pattern B sequence starts with 1 at k+1.
        # If (k+1) is odd, pattern A sequence starts with 1 at k+1.
        if k + 2 < N: # If suffix exists
            if (k + 1) % 2 == 0: # T[k+1]=1 matches pattern B at even k+1
                current_cost_11 += suff_B_cost[k+2]
            else: # T[k+1]=1 matches pattern A at odd k+1
                current_cost_11 += suff_A_cost[k+2]
        
        min_total_cost = min(min_total_cost, current_cost_11)

    sys.stdout.write(str(min_total_cost) + "
")

solve()