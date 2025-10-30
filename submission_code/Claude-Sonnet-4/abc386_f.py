# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

m, n = len(S), len(T)

# If the length difference is already > K, impossible
if abs(m - n) > K:
    print("No")
else:
    # DP table
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S[i-1] == T[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # delete
                    dp[i][j-1],    # insert
                    dp[i-1][j-1]   # replace
                )
            
            # Optimization: if value exceeds K, we can cap it
            if dp[i][j] > K:
                dp[i][j] = K + 1
    
    if dp[m][n] <= K:
        print("Yes")
    else:
        print("No")