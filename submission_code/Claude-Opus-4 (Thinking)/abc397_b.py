# YOUR CODE HERE
S = input()
n = len(S)

# dp[i][j] = min insertions to use first i chars of S to fill first j positions of pattern
INF = float('inf')
max_len = 2 * n  # Upper bound on the final length
dp = [[INF] * (max_len + 1) for _ in range(n + 1)]

dp[0][0] = 0

for i in range(n + 1):
    for j in range(max_len + 1):
        if dp[i][j] == INF:
            continue
        
        # Insert a character at position j+1
        if j + 1 <= max_len:
            dp[i][j+1] = min(dp[i][j+1], dp[i][j] + 1)
        
        # Use S[i] at position j+1
        if i < n and j + 1 <= max_len:
            pattern_char = 'i' if (j + 1) % 2 == 1 else 'o'
            if S[i] == pattern_char:
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j])

# Find the minimum insertions for any even length that uses all characters of S
ans = INF
for j in range(2, max_len + 1, 2):  # Even lengths only
    ans = min(ans, dp[n][j])

print(ans)