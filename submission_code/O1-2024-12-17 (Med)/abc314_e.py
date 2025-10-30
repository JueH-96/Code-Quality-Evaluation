def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    costs = []
    p = []
    rewards = []
    
    idx = 2
    for _ in range(N):
        C_i = int(input_data[idx]); idx += 1
        P_i = int(input_data[idx]); idx += 1
        S_i = list(map(int, input_data[idx:idx+P_i]))
        idx += P_i
        
        costs.append(C_i)
        p.append(P_i)
        rewards.append(S_i)
    
    # dp[x] = minimum expected cost to reach at least M points starting with x points
    # We'll store dp for x=0..M; dp[M] = 0 (once we have M or more, cost needed is 0).
    dp = [0.0]*(M+1)
    
    # Value-iteration-like approach (since M <= 100, a simple iteration is feasible)
    # Recurrence:
    # dp[x] = min_i [ C_i + (1/P_i)*sum_{j=1..P_i}( dp[min(M, x+S_{i,j})] ) ] for x < M
    # dp[x] = 0 for x >= M
    
    # We iterate until values converge
    MAX_ITER = 600
    EPS = 1e-10
    
    for _ in range(MAX_ITER):
        new_dp = [0.0]*(M+1)
        
        # Precompute partial sums for each wheel i and for each x in [0..M-1]
        # partial_sum[i][x] = sum of dp[x + S_{i,j}] (clamped at M) over j
        partial_sum = []
        for i in range(N):
            arr = [0.0]*M
            for x in range(M):
                s = 0.0
                for r in rewards[i]:
                    nxt = x + r
                    if nxt > M:
                        nxt = M
                    s += dp[nxt]
                arr[x] = s
            partial_sum.append(arr)
        
        max_diff = 0.0
        
        # Update dp values for x in [0..M-1]
        for x in range(M):
            best = float('inf')
            for i in range(N):
                # cost for playing wheel i
                cost_i = costs[i]
                # expected cost from new points
                exp_next = partial_sum[i][x] / p[i]
                val = cost_i + exp_next
                if val < best:
                    best = val
            new_dp[x] = best
            diff = abs(new_dp[x] - dp[x])
            if diff > max_diff:
                max_diff = diff
        
        dp = new_dp
        # dp[M] stays 0 by definition
        dp.append(0.0)  # ensure index M exists and is 0
        dp[M] = 0.0
        
        if max_diff < EPS:
            break
    
    # The answer is dp[0]
    print(dp[0])

# Do not forget to call main()
if __name__ == "__main__":
    main()