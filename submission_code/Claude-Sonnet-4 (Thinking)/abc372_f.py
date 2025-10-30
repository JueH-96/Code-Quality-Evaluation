MOD = 998244353

N, M, K = map(int, input().split())

# Store additional edges
additional_edges = []
for _ in range(M):
    x, y = map(int, input().split())
    additional_edges.append((x, y))

# DP
prev_dp = [0] * (N+1)
prev_dp[1] = 1

for i in range(K):
    curr_dp = [0] * (N+1)
    
    # Handle regular edges: j gets contribution from (j-1) or N if j=1
    for j in range(1, N+1):
        if j == 1:
            curr_dp[j] = prev_dp[N]
        else:
            curr_dp[j] = prev_dp[j-1]
    
    # Handle additional edges
    for x, y in additional_edges:
        curr_dp[y] = (curr_dp[y] + prev_dp[x]) % MOD
    
    prev_dp = curr_dp

result = sum(prev_dp[j] for j in range(1, N+1)) % MOD
print(result)