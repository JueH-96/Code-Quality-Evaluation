N, Q = map(int, input().split())
operations = []
for _ in range(Q):
    p, v = map(int, input().split())
    operations.append((p, v))

MOD = 998244353

# DP approach
# dp[state] = number of ways to reach this state
dp = {tuple([0] * N): 1}

for i in range(Q):
    p, v = operations[i]
    new_dp = {}
    
    for state, count in dp.items():
        # Option 1: Set S[0..p-1] to v (1-indexed: S[1..p])
        if all(state[j] <= v for j in range(p)):
            new_state = list(state)
            for j in range(p):
                new_state[j] = v
            new_state = tuple(new_state)
            new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
        
        # Option 2: Set S[p-1..N-1] to v (1-indexed: S[p..N])
        if all(state[j] <= v for j in range(p - 1, N)):
            new_state = list(state)
            for j in range(p - 1, N):
                new_state[j] = v
            new_state = tuple(new_state)
            new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
    
    dp = new_dp

# Sum all states after Q operations
result = sum(dp.values()) % MOD
print(result)