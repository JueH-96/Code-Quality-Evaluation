# YOUR CODE HERE
import sys

def main():
    """
    This is the main function that reads input, solves the problem, and prints the output.
    """
    # Using 0-based indexing for parts internally.
    # Input is 1-based.
    
    # Reading input
    try:
        input = sys.stdin.readline
        N_str, Q_str = input().split()
        N, Q = int(N_str), int(Q_str)
        queries = []
        for _ in range(Q):
            h, t_str = input().split()
            queries.append((h, int(t_str) - 1))
    except (IOError, ValueError):
        return

    inf = float('inf')

    def cost(p1, p2, n):
        """Calculates the shortest distance between two parts on the ring."""
        diff = abs(p1 - p2)
        return min(diff, n - diff)

    # dp[j] = minimum total cost with the "other" hand at position j.
    # The "other" hand is the one not specified in the current instruction.
    dp = [inf] * N

    # Initial state: Left hand at part 1 (idx 0), Right hand at part 2 (idx 1).
    pos_L, pos_R = 0, 1
    
    # Process the first instruction to initialize the DP table.
    h, t = queries[0]
    if h == 'L':
        # Left hand moves from pos_L to t.
        # The "other" hand (Right) moves from pos_R to j.
        for j in range(N):
            if j == t:
                continue
            dp[j] = cost(pos_L, t, N) + cost(pos_R, j, N)
    else:  # h == 'R'
        # Right hand moves from pos_R to t.
        # The "other" hand (Left) moves from pos_L to j.
        for j in range(N):
            if j == t:
                continue
            dp[j] = cost(pos_R, t, N) + cost(pos_L, j, N)

    prev_h, prev_t = queries[0]

    # Process remaining instructions
    for i in range(1, Q):
        curr_h, curr_t = queries[i]
        
        prev_dp = list(dp)
        prev_dp[prev_t] = inf
        
        dp = [inf] * N

        if prev_h == curr_h:
            # Same hand specified. Roles do not swap.
            # We need to compute: cost(prev_t, curr_t) + min_k(prev_dp[k] + cost(k, j))
            
            B = [prev_dp[k] - k for k in range(N)]
            C = [prev_dp[k] + k for k in range(N)]
            
            P_B = [inf] * N; P_B[0] = B[0]
            for k in range(1, N): P_B[k] = min(P_B[k-1], B[k])

            S_B = [inf] * N; S_B[N-1] = B[N-1]
            for k in range(N-2, -1, -1): S_B[k] = min(S_B[k+1], B[k])

            P_C = [inf] * N; P_C[0] = C[0]
            for k in range(1, N): P_C[k] = min(P_C[k-1], C[k])
            
            S_C = [inf] * N; S_C[N-1] = C[N-1]
            for k in range(N-2, -1, -1): S_C[k] = min(S_C[k+1], C[k])

            M = [inf] * N
            for j in range(N):
                # min_k (prev_dp[k] + (j-k+N)%N)
                val_L = j + min(P_B[j], (S_B[j+1] if j+1 < N else inf) + N)
                # min_k (prev_dp[k] + (k-j+N)%N)
                val_R = -j + min(S_C[j], (P_C[j-1] if j-1 >= 0 else inf) + N)
                M[j] = min(val_L, val_R)
            
            cost_move_spec = cost(prev_t, curr_t, N)
            for j in range(N):
                if j == curr_t:
                    continue
                dp[j] = M[j] + cost_move_spec
        else:
            # Different hands specified. Roles swap.
            # We need: cost(prev_t, j) + min_k(prev_dp[k] + cost(k, curr_t))

            min_cost_other_hand = inf
            for k in range(N):
                min_cost_other_hand = min(min_cost_other_hand, prev_dp[k] + cost(k, curr_t, N))

            for j in range(N):
                if j == curr_t:
                    continue
                cost_move_prev_spec = cost(prev_t, j, N)
                dp[j] = min_cost_other_hand + cost_move_prev_spec
        
        prev_h, prev_t = curr_h, curr_t

    ans = min(dp)
    print(ans)

if __name__ == "__main__":
    main()