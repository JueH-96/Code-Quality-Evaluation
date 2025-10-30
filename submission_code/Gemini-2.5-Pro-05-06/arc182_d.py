import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # dp_prev[k_idx] stores min cost for prefix A[:i], with A[i-1] using k_val = k_idx - N.
    # dp_curr[k_idx] stores min cost for prefix A[:i+1], with A[i] using k_val = k_idx - N.
    # k_val is the number of "net wraparounds" for an element.
    
    # The size of the DP array needs to accommodate k_val from -N to N.
    # (Actually, for A[i], k ranges from -i to i relative to k for A[-1]).
    # So, indices from 0 to 2*N. Size is 2*N + 1.
    # Index N corresponds to k_val = 0.
    dp_arr_size = 2 * N + 1 # Max index is 2*N.
    
    # Initialize dp_prev for a hypothetical element A[-1] which has cost 0 and k_val = 0.
    dp_prev = [float('inf')] * dp_arr_size
    dp_prev[N] = 0

    # Iterate through each element from A[0] to A[N-1] (index i in A/B)
    for i in range(N):
        dp_curr = [float('inf')] * dp_arr_size
        
        # Valid range for k_val for A[i] is [-i, i] (assuming k_A[-1]=0).
        # So k_idx for A[i] is [N-i, N+i].
        # Loop bounds for k_curr_idx:
        min_idx_loop = N - i
        max_idx_loop = N + i

        # Iterate over possible k_val for current element A[i]
        for k_curr_idx in range(min_idx_loop, max_idx_loop + 1):
            k_curr_val = k_curr_idx - N
            
            cost_for_this_element = abs((B[i] + k_curr_val * M) - A[i])
            
            min_prev_total_cost = float('inf')

            # Option 1: k_prev_val = k_curr_val - 1 (k_prev_idx = k_curr_idx - 1)
            # Check array bounds: k_curr_idx - 1 must be >= 0
            if k_curr_idx - 1 >= 0:
                 min_prev_total_cost = min(min_prev_total_cost, dp_prev[k_curr_idx - 1])
            
            # Option 2: k_prev_val = k_curr_val (k_prev_idx = k_curr_idx)
            # k_curr_idx is always within [0, 2N] by loop construction
            min_prev_total_cost = min(min_prev_total_cost, dp_prev[k_curr_idx])
            
            # Option 3: k_prev_val = k_curr_val + 1 (k_prev_idx = k_curr_idx + 1)
            # Check array bounds: k_curr_idx + 1 must be < dp_arr_size
            if k_curr_idx + 1 < dp_arr_size:
                 min_prev_total_cost = min(min_prev_total_cost, dp_prev[k_curr_idx + 1])
            
            if min_prev_total_cost != float('inf'):
                dp_curr[k_curr_idx] = min_prev_total_cost + cost_for_this_element
        
        dp_prev = dp_curr

    ans = float('inf')
    # After processing A[N-1] (i=N-1 loop), dp_prev holds costs for A[N-1].
    # Relevant k_val for A[N-1] are [-(N-1), N-1].
    # So k_idx are [N-(N-1), N+(N-1)] = [1, 2N-1].
    # (If N=1, k_val for A[0] is 0, so k_idx is N. min_idx_final = max_idx_final = N)
    # Problem constraints N >= 2.
    min_k_idx_final = N - (N - 1) 
    max_k_idx_final = N + (N - 1)
    
    for k_idx in range(min_k_idx_final, max_k_idx_final + 1):
        ans = min(ans, dp_prev[k_idx])
            
    print(ans)

solve()