# YOUR CODE HERE
def min_operations(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                   dp[i][j-1],    # insertion
                                   dp[i-1][j-1])  # replacement
    
    return dp[m][n]

K = int(input())
S = input().strip()
T = input().strip()

operations_needed = min_operations(S, T)

if operations_needed <= K:
    print("Yes")
else:
    print("No")