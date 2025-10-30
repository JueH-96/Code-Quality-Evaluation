n = int(input())
a = list(map(int, input().split()))

# dp[i][j] = maximum score from reducing A[i:j+1] to at most 1 element
dp = [[0] * n for _ in range(n)]

# Fill DP table
for length in range(2, n + 1):  # length of subarray
    for i in range(n - length + 1):
        j = i + length - 1
        
        if length == 2:
            dp[i][j] = abs(a[i] - a[j])
        else:
            # Try all possible ways to split [i, j] into two parts
            for k in range(i, j):
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k+1][j])

print(dp[0][n-1])