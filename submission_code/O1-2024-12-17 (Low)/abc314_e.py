def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    M = int(input_data[1])
    idx = 2
    costs = []
    wheels = []
    for _ in range(N):
        C_i = int(input_data[idx]); idx += 1
        P_i = int(input_data[idx]); idx += 1
        S_i = list(map(int, input_data[idx:idx+P_i]))
        idx += P_i
        costs.append(C_i)
        wheels.append(S_i)
    
    # DP[x] = minimal expected cost to achieve at least M points starting from x points
    # For x >= M, DP[x] = 0
    # For x < M, DP[x] = min over i of [ C_i + 1/P_i * sum(DP[x + S_{i,j} (clamped to M)] ) ]
    
    # We'll store DP for states 0..M. DP[M] = 0 exactly.
    # We'll use iterative approach to solve until convergence.
    
    dp = [0.0]*(M+1)  # dp[x] -> cost
    # We know dp[M] = 0 and we will iteratively update dp[0..M-1].
    
    # Precompute for each wheel i, the average transitions for each x.
    # However, we can simply do it in each iteration directly.
    
    # We do iterative approach until values stabilize.
    
    # A function to compute the cost of playing wheel i from state x:
    #    cost_i(x) = C_i + (1/P_i) * sum_{j=1..P_i} dp[min(M, x + S_{i,j})]
    
    MAX_ITER = 10000
    EPS = 1e-14
    
    for _ in range(MAX_ITER):
        updated_dp = dp[:]
        max_diff = 0.0
        for x in range(M-1, -1, -1):
            # compute best play from x
            best = float('inf')
            for i in range(N):
                C_i = costs[i]
                S_i = wheels[i]
                P_i = len(S_i)
                # expected next cost
                tmp = 0.0
                for s in S_i:
                    nxt = x + s
                    if nxt >= M:
                        nxt = M
                    tmp += dp[nxt]
                tmp = tmp / P_i  # average next cost
                cost_i = C_i + tmp
                if cost_i < best:
                    best = cost_i
            updated_dp[x] = best
            diff = abs(best - dp[x])
            if diff > max_diff:
                max_diff = diff
        dp = updated_dp
        if max_diff < EPS:
            break
    
    print(f"{dp[0]:.9f}")

# Do not forget to call main()
if __name__ == "__main__":
    main()