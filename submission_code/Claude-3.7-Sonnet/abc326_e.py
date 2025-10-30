N = int(input())
A = list(map(int, input().split()))

MOD = 998244353

# dp[i] represents the expected earnings if Aoki starts with x = i
dp = [0] * (N + 1)

# Calculate the modular inverse of N
inv_N = pow(N, MOD-2, MOD)

# Calculate the term (1 + 1/N) used in the recurrence
one_plus_inv_N = (1 + inv_N) % MOD

# Calculate dp[N-1], dp[N-2], ..., dp[0] using the optimized recurrence
for i in range(N, 0, -1):
    # The term A_i / N
    A_term = (A[i-1] * inv_N) % MOD
    
    # The term dp[i] * (1 + 1/N)
    dp_term = (dp[i] * one_plus_inv_N) % MOD
    
    # Calculate dp[i-1] = A_term + dp_term
    dp[i-1] = (A_term + dp_term) % MOD

print(dp[0])