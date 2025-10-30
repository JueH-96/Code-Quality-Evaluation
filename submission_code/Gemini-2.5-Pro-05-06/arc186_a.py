import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())

    # dp[i] will store the set of possible numbers of fixed cells for an i x i matrix
    dp = [set() for _ in range(N + 1)]

    # Base case S_0
    if N >= 0:
        dp[0] = {0}

    # Iteratively compute S_n for n = 1 to N
    for i in range(1, N + 1):
        # Add i*i (all cells fixed, e.g., all-0s matrix)
        current_s_i = {i*i}
        
        # Add 0 if i >= 2 (no cells fixed, e.g., identity matrix)
        if i >= 2:
            current_s_i.add(0)
        
        # For prev_n from 0 to i-1:
        # For k_prev in S_prev_n:
        #   Add (i^2 - prev_n^2) + k_prev to S_i
        # This construction corresponds to making an i x i matrix where an outer "frame"
        # of i^2 - prev_n^2 cells is fixed (e.g., all 1s), and the inner
        # prev_n x prev_n submatrix B has k_prev fixed cells.
        for prev_n in range(i): # prev_n from 0 to i-1
            term_fixed_frame = i*i - prev_n*prev_n
            for k_prev in dp[prev_n]:
                current_s_i.add(term_fixed_frame + k_prev)
        dp[i] = current_s_i
        
    possible_K = dp[N]
    
    results = []
    for _ in range(Q):
        K_i = int(sys.stdin.readline())
        if K_i in possible_K:
            results.append("Yes")
        else:
            results.append("No")
            
    sys.stdout.write("
".join(results) + "
")

solve()