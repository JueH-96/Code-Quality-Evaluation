def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    MOD = 998244353
    
    S = sys.stdin.readline().strip()
    n = len(S)

    # dp[i][b] = number of ways to parse the first i chars with 'b' open parentheses
    dp = [[0]*(n+1) for _ in range(n+1)]
    dp[0][0] = 1
    
    for i in range(n):
        for b in range(i+1):
            if dp[i][b] == 0:
                continue
            ways = dp[i][b]
            # If current char can be '('
            if S[i] in ('(', '?'):
                dp[i+1][b+1] = (dp[i+1][b+1] + ways) % MOD
            # If current char can be ')'
            if (S[i] in (')', '?')) and b > 0:
                dp[i+1][b-1] = (dp[i+1][b-1] + ways) % MOD
    
    print(dp[n][0] % MOD)

if __name__ == "__main__":
    main()