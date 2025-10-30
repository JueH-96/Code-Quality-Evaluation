mod = 998244353

# Read input
N, X = map(int, input().split())
T = list(map(int, input().split()))
T1 = T[0]

a = max(0, X - T1 + 1)
b = X

# Compute inverse of N modulo mod
inv_N = pow(N, mod-2, mod)

# Initialize DP
dp = [0] * (X + 1)
dp[0] = 1

# Fill DP
for s in range(1, X + 1):
    for t in T:
        if t > s:
            continue
        dp[s] = (dp[s] + dp[s - t] * inv_N) % mod

# Compute the sum of probabilities for a <= s <= b
sum_s = 0
for s in range(a, b + 1):
    sum_s = (sum_s + dp[s]) % mod

# Multiply by inv_N to get the final probability
result = sum_s * inv_N % mod

print(result)