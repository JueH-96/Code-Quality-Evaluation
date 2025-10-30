N, M, K = map(int, input().split())
A = list(map(int, input().split()))

MOD = 998244353

# dp[(x, r)] = number of subsets with XOR = x and size ≡ r (mod M)
dp = {(0, 0): 1}  # empty subset

for a in A:
    new_dp = {}
    for (x, r), count in dp.items():
        # Don't include a
        key = (x, r)
        new_dp[key] = (new_dp.get(key, 0) + count) % MOD
        # Include a
        new_key = (x ^ a, (r + 1) % M)
        new_dp[new_key] = (new_dp.get(new_key, 0) + count) % MOD
    dp = new_dp

# Calculate the answer
ans = 0
for (x, r), count in dp.items():
    if r == 0 and x != 0:  # size divisible by M, and XOR != 0
        ans = (ans + count * pow(x, K, MOD)) % MOD

print(ans)