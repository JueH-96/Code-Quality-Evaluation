n = int(input())
A = list(map(int, input().split()))

if n <= 1:
    print(0)
    exit()

dp = [[0] * n for _ in range(n)]

# Base case: adjacent pairs
for i in range(n - 1):
    dp[i][i + 1] = abs(A[i] - A[i + 1])

# Fill DP table for lengths >= 3
for length in range(3, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        
        if length % 2 == 1:  # Odd length, one element will remain
            for k in range(i, j + 1):
                score = 0
                if k > i:
                    score += dp[i][k - 1]
                if k < j:
                    score += dp[k + 1][j]
                dp[i][j] = max(dp[i][j], score)
        else:  # Even length, all elements will be removed
            # Option 1: pair first and last elements
            dp[i][j] = abs(A[i] - A[j]) + dp[i + 1][j - 1]
            
            # Option 2: split into two parts with even length
            for k in range(i + 1, j, 2):  # k-i is always odd (even left part)
                if (j - k) % 2 == 0:  # Check if right part has even length
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k + 1][j])

print(dp[0][n - 1])