n = int(input())
a = list(map(int, input().split()))

if n == 1:
    print(0)
elif n == 2:
    print(abs(a[0] - a[1]))
else:
    # dp[i][j] = max score removing all elements from i to j inclusive
    # only valid when j - i + 1 is even
    dp = [[0] * n for _ in range(n)]
    
    # Base case: pairs
    for i in range(n - 1):
        dp[i][i + 1] = abs(a[i] - a[i + 1])
    
    # Fill dp table for even-length intervals
    for length in range(4, n + 1, 2):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Try all possible pairings for a[i]
            best = 0
            for k in range(i + 1, j + 1, 2):
                score = abs(a[i] - a[k])
                
                # Add score from removing elements between i and k
                if k > i + 1:
                    score += dp[i + 1][k - 1]
                
                # Add score from removing elements after k
                if k < j:
                    score += dp[k + 1][j]
                
                best = max(best, score)
            
            dp[i][j] = best
    
    if n % 2 == 0:
        print(dp[0][n - 1])
    else:
        # Find best element to leave (must be at even index)
        max_score = 0
        for k in range(0, n, 2):
            score = 0
            if k > 0:
                score += dp[0][k - 1]
            if k < n - 1:
                score += dp[k + 1][n - 1]
            max_score = max(max_score, score)
        print(max_score)