import sys

def count_lcs(s, m):
    n = len(s)
    mod = 998244353
    
    # Precompute the number of strings of length m
    num_strings = pow(26, m, mod)
    
    # Initialize the dp table
    dp = [[0] * (n+1) for _ in range(n+1)]
    
    # Fill the dp table
    for i in range(1, n+1):
        for j in range(1, n+1):
            if s[i-1] == s[j-1]:
                dp[i][j] = (dp[i-1][j-1] + 1) % mod
            else:
                dp[i][j] = (dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]) % mod
    
    # Compute the answers
    ans = []
    for k in range(n+1):
        count = pow(26, m-k, mod)
        ans.append(count * dp[n][k] % mod)
    
    return ans

# Read the input
n, m = map(int, input().split())
s = input()

# Solve the problem
ans = count_lcs(s, m)

# Print the output
print(" ".join(map(str, ans)))