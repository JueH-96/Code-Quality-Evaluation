MOD = 998244353

K = int(input())
C = list(map(int, input().split()))

# Precompute factorials and their modular inverses
fact = [1] * (K + 1)
for i in range(1, K + 1):
    fact[i] = (fact[i-1] * i) % MOD

inv_fact = [1] * (K + 1)
if K > 0:
    inv_fact[K] = pow(fact[K], MOD - 2, MOD)
    for i in range(K - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

def mod_comb(n, k):
    if k > n or k < 0:
        return 0
    return (fact[n] * inv_fact[k] % MOD) * inv_fact[n - k] % MOD

# dp[i][j] = number of strings of length j using first i letters
dp = [[0] * (K + 1) for _ in range(27)]

# Base case
dp[0][0] = 1

for i in range(1, 27):  # for each letter (1-indexed)
    for j in range(K + 1):  # for each length
        for k in range(min(C[i-1], j) + 1):  # use k occurrences of letter i
            comb = mod_comb(j, k)
            dp[i][j] = (dp[i][j] + comb * dp[i-1][j-k]) % MOD

# Answer is sum of dp[26][1] + dp[26][2] + ... + dp[26][K]
ans = 0
for j in range(1, K + 1):
    ans = (ans + dp[26][j]) % MOD

print(ans)