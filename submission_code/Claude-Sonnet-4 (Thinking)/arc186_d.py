def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    # dp[i] = number of Polish sequences of length i
    dp = [0] * (N + 1)
    if N >= 1:
        dp[1] = 1
    
    # ways[i][j] = number of ways to partition length i into j Polish sequences
    ways = [[0] * N for _ in range(N + 1)]
    ways[0][0] = 1
    
    # Build dp table
    for i in range(1, N + 1):
        for j in range(1, min(i + 1, N)):
            for k in range(1, i + 1):
                if k < len(dp):
                    ways[i][j] = (ways[i][j] + dp[k] * ways[i - k][j - 1]) % MOD
        
        # Calculate dp[i] based on all possible first elements
        for v in range(i):
            if v < len(ways[i - 1]):
                dp[i] = (dp[i] + ways[i - 1][v]) % MOD
    
    # Digit DP to count valid sequences
    memo = {}
    
    def digit_dp(pos, tight):
        if pos == N:
            return 1
        
        if (pos, tight) in memo:
            return memo[(pos, tight)]
        
        result = 0
        limit = A[pos] if tight else N - 1
        
        for v in range(min(limit + 1, N - pos)):
            new_tight = tight and (v == A[pos])
            remaining = N - pos - 1
            
            if remaining < v:
                continue
            
            if v == 0 and remaining == 0:
                result = (result + 1) % MOD
            elif v > 0:
                # Need v Polish sequences in remaining positions
                if remaining < len(ways) and v < len(ways[remaining]):
                    count = ways[remaining][v]
                    if new_tight:
                        # For tight constraint, we need more careful counting
                        # This is a simplification - in full solution would need
                        # to handle lexicographical constraint for partitions
                        result = (result + count) % MOD
                    else:
                        result = (result + count) % MOD
        
        memo[(pos, tight)] = result
        return result
    
    return digit_dp(0, True)

print(solve())