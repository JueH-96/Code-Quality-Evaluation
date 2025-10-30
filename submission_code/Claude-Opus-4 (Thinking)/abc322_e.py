N, K, P = map(int, input().split())
plans = []
for _ in range(N):
    line = list(map(int, input().split()))
    C = line[0]
    A = line[1:]
    plans.append((C, A))

# DP with states: dp[i][state] where state is a tuple of parameter values capped at P
dp = [{} for _ in range(N + 1)]
dp[0][(0,) * K] = 0

for i in range(1, N + 1):
    C, A = plans[i - 1]
    
    # Copy values from dp[i-1] (not using plan i)
    dp[i] = dp[i - 1].copy()
    
    # Try using plan i
    for state, cost in dp[i - 1].items():
        new_state = tuple(min(state[j] + A[j], P) for j in range(K))
        if new_state not in dp[i] or dp[i][new_state] > cost + C:
            dp[i][new_state] = cost + C

# Check if the goal is achieved
goal_state = (P,) * K
if goal_state in dp[N]:
    print(dp[N][goal_state])
else:
    print(-1)