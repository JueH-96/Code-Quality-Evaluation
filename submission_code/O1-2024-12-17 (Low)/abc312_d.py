def main():
    import sys
    MOD = 998244353
    
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # dp[i][j] = number of ways to parse the first i characters where j is the number
    # of currently "open" parentheses that haven't been closed yet.
    # We'll compute dp[n][0] in the end.
    dp = [[0]*(n+1) for _ in range(n+1)]
    
    dp[0][0] = 1  # base case: 0 chars used, 0 open parentheses => 1 way
    
    for i in range(n):
        c = S[i]
        for open_count in range(n):
            if dp[i][open_count] == 0:
                continue
            
            ways = dp[i][open_count]
            
            # If we can interpret S[i] as '(':
            if c == '(' or c == '?':
                dp[i+1][open_count+1] = (dp[i+1][open_count+1] + ways) % MOD
            
            # If we can interpret S[i] as ')' (and open_count > 0):
            if (c == ')' or c == '?') and open_count > 0:
                dp[i+1][open_count-1] = (dp[i+1][open_count-1] + ways) % MOD
    
    print(dp[n][0] % MOD)

# Call main() to initiate the solution.
if __name__ == "__main__":
    main()