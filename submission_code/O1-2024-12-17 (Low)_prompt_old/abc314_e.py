def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    idx = 0
    
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1
    
    wheels = []
    for _ in range(N):
        C = int(input_data[idx]); idx += 1
        P = int(input_data[idx]); idx += 1
        S = list(map(int, input_data[idx:idx+P]))
        idx += P
        wheels.append((C, P, S))
    
    # dp[x] will represent the minimal expected cost to go from x points to at least M points.
    # dp[M] = 0 because if we already have M or more points, no more cost is needed.
    # For x < M: dp[x] = min over i in [1..N] of (C_i + (1/P_i) * sum_{j=1..P_i} dp[min(M, x + S_{i,j})])
    
    # We'll solve by iterative approximation until convergence.
    
    dp = [0.0]*(M+1)  # dp[0..M], dp[M] = 0
    
    # A function to perform one update iteration
    def update_dp(dp_old):
        dp_new = dp_old[:]
        for x in range(M-1, -1, -1):
            best = float('inf')
            for (C, P, S_list) in wheels:
                # sum up dp of the new states
                temp_sum = 0.0
                for s in S_list:
                    nx = x + s
                    if nx >= M:
                        nx = M
                    temp_sum += dp_old[nx]
                exp_cost = C + (temp_sum / P)
                if exp_cost < best:
                    best = exp_cost
            dp_new[x] = best
        return dp_new
    
    # Iterate until difference is small
    import math
    
    # Initialize dp array (we can guess a large number for dp[x] initially, but 0 is also ok)
    for x in range(M):
        dp[x] = 0.0  # just a starting guess
    
    # Perform iterations
    EPS = 1e-14
    for _ in range(100000):
        new_dp = update_dp(dp)
        # Check max difference
        diff = 0.0
        for i in range(M+1):
            diff = max(diff, abs(new_dp[i] - dp[i]))
        dp = new_dp
        if diff < EPS:
            break
    
    # dp[0] is the answer
    print(f"{dp[0]:.9f}")    

# Let's call solve() to run the solution.
# (The testing environment will call solve() again, but we can call it here safely.)
if __name__ == "__main__":
    solve()