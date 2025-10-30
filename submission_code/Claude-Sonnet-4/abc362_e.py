# YOUR CODE HERE
MOD = 998244353

n = int(input())
a = list(map(int, input().split()))

result = []

# For k = 1, every element forms an arithmetic sequence
result.append(n)

if n == 1:
    print(1)
else:
    # For k = 2, every pair forms an arithmetic sequence
    result.append(n * (n - 1) // 2)
    
    # For k >= 3, use DP
    # dp[i][j] = number of arithmetic subsequences ending at positions i and j
    dp = [[0] * n for _ in range(n)]
    
    # Initialize: every pair can start an arithmetic sequence of length 2
    for i in range(n):
        for j in range(i + 1, n):
            dp[i][j] = 1
    
    # For each length k from 3 to n
    for k in range(3, n + 1):
        new_dp = [[0] * n for _ in range(n)]
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if dp[i][j] > 0:
                    # Try to extend the arithmetic sequence ending at i, j
                    diff = a[j] - a[i]
                    next_val = a[j] + diff
                    
                    # Look for the next element
                    for l in range(j + 1, n):
                        if a[l] == next_val:
                            new_dp[j][l] = (new_dp[j][l] + dp[i][j]) % MOD
        
        # Count total subsequences of length k
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total = (total + new_dp[i][j]) % MOD
        
        result.append(total)
        dp = new_dp
    
    print(' '.join(map(str, result)))