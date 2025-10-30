N, M, K = map(int, input().split())
A = list(map(int, input().split()))

MOD = 998244353

# dp[r] = dictionary mapping XOR value to count of subsequences with length ≡ r (mod M)
dp = [{} for _ in range(M)]
dp[0][0] = 1  # empty subsequence

for a in A:
    new_dp = [{} for _ in range(M)]
    for r in range(M):
        for x, count in dp[r].items():
            # Don't include a
            new_dp[r][x] = new_dp[r].get(x, 0) + count
            new_dp[r][x] %= MOD
            
            # Include a
            new_r = (r + 1) % M
            new_x = x ^ a
            new_dp[new_r][new_x] = new_dp[new_r].get(new_x, 0) + count
            new_dp[new_r][new_x] %= MOD
    
    dp = new_dp

ans = 0
for x, count in dp[0].items():
    if x == 0:
        # Exclude empty subsequence
        count -= 1
    if count > 0:
        ans = (ans + count * pow(x, K, MOD)) % MOD

print(ans)