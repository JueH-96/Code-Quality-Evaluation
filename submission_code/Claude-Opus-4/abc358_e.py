# YOUR CODE HERE
MOD = 998244353

K = int(input())
C = list(map(int, input().split()))

# dp[length] = number of strings of exactly that length
dp = [0] * (K + 1)
dp[0] = 1  # empty string

# For each letter
for i in range(26):
    if C[i] == 0:
        continue
    
    # Process lengths in reverse to avoid using updated values
    for length in range(K, 0, -1):
        # Try using 1 to min(C[i], length) copies of this letter
        for use in range(1, min(C[i], length) + 1):
            if length - use >= 0:
                dp[length] = (dp[length] + dp[length - use]) % MOD

# Sum up all lengths from 1 to K
result = sum(dp[1:K+1]) % MOD
print(result)