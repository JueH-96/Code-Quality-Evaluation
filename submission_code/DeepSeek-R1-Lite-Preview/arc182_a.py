def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    N, Q = map(int, sys.stdin.readline().split())
    operations = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    
    # Initialize prefix and suffix maxima
    prefix_max = [0] * (N + 1)
    suffix_max = [0] * (N + 1)
    
    # DP array: dp[i] is the number of ways to perform first i operations
    dp = [0] * (Q + 1)
    dp[0] = 1  # There's one way to perform zero operations
    
    for i in range(1, Q + 1):
        P, V = operations[i - 1]
        
        # Check if choosing to set prefix P to V is possible
        # The current prefix max up to P must be <= V
        current_prefix_max = max(prefix_max[1:P+1])
        if current_prefix_max <= V:
            dp[i] = (dp[i] + dp[i - 1]) % MOD
            # Update prefix_max for the next operations
            for j in range(1, P + 1):
                prefix_max[j] = max(prefix_max[j], V)
        
        # Check if choosing to set suffix P to V is possible
        # The current suffix max from P to N must be <= V
        current_suffix_max = max(suffix_max[P:N+1])
        if current_suffix_max <= V:
            dp[i] = (dp[i] + dp[i - 1]) % MOD
            # Update suffix_max for the next operations
            for j in range(P, N + 1):
                suffix_max[j] = max(suffix_max[j], V)
    
    print(dp[Q] % MOD)

if __name__ == "__main__":
    main()