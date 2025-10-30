def solve():
    MOD = 998244353
    
    # Read input
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    
    # dp[i][mask] represents number of sequences of length i that don't contain
    # subsequences represented by mask as a state of matching X
    dp = [[0] * (1 << M) for _ in range(N + 1)]
    dp[0][0] = 1
    
    # For each position in sequence
    for i in range(N):
        # For each current state of matching
        for mask in range(1 << M):
            if dp[i][mask] == 0:
                continue
            
            # Try each possible number at current position
            for k in range(1, K + 1):
                # Calculate new state after adding k
                new_mask = mask
                for j in range(M):
                    # If j-th bit is not set and current number matches X[j]
                    # and all previous required matches are set
                    if ((mask >> j) & 1) == 0 and k == X[j]:
                        # Check if all previous positions are matched
                        if j == 0 or ((mask >> (j-1)) & 1) == 1:
                            new_mask |= (1 << j)
                
                # If new_mask doesn't represent complete match of X
                if new_mask != (1 << M) - 1:
                    dp[i + 1][new_mask] = (dp[i + 1][new_mask] + dp[i][mask]) % MOD
    
    # Sum all states except the one representing complete match of X
    result = 0
    for mask in range(1 << M):
        result = (result + dp[N][mask]) % MOD
    
    print(result)

solve()