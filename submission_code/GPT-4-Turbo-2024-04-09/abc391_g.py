import sys
input = sys.stdin.read
MOD = 998244353

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Total number of strings of length M
    total_strings = pow(26, M, MOD)
    
    # Dynamic programming table
    # dp[i][j] will store the number of strings of length j that have an LCS of exactly i with S[:i]
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1  # 1 way to have an LCS of length 0 with an empty string and empty string
    
    # Fill the dp table
    for i in range(N + 1):
        for j in range(M + 1):
            if dp[i][j] == 0:
                continue
            
            # If we add a character not in S to the string of length j
            if j + 1 <= M:
                dp[i][j + 1] = (dp[i][j + 1] + dp[i][j] * (26 - (1 if i < N else 0))) % MOD
            
            # If we add the next character of S to the string of length j
            if i < N and j + 1 <= M:
                dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
    
    # Calculate the results for each k
    results = [0] * (N + 1)
    for k in range(N + 1):
        for j in range(M + 1):
            results[k] = (results[k] + dp[k][j]) % MOD
    
    # Calculate the number of strings with LCS length exactly k
    # We need to subtract the number of strings with LCS length > k from total_strings
    ans = [0] * (N + 1)
    for k in range(N + 1):
        if k == 0:
            ans[k] = total_strings
            for l in range(1, N + 1):
                ans[k] = (ans[k] - results[l] + MOD) % MOD
        else:
            ans[k] = results[k]
    
    # Print the answers
    print(" ".join(map(str, ans)))