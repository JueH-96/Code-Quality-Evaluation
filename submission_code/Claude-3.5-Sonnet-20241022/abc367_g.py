def solve():
    MOD = 998244353
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # dp[i][j][xor] = number of subsequences ending at i with length mod M = j and XOR = xor
    dp = [{} for _ in range(M)]
    
    # Initialize with empty sequence
    dp[0][0] = 1
    
    # For each number in sequence
    for x in A:
        # Make copy of current state
        new_dp = [{} for _ in range(M)]
        for j in range(M):
            for xor in dp[j]:
                # Don't take current number
                if xor in new_dp[j]:
                    new_dp[j][xor] = (new_dp[j][xor] + dp[j][xor]) % MOD
                else:
                    new_dp[j][xor] = dp[j][xor]
                
                # Take current number
                next_j = (j + 1) % M
                next_xor = xor ^ x
                if next_xor in new_dp[next_j]:
                    new_dp[next_j][next_xor] = (new_dp[next_j][next_xor] + dp[j][xor]) % MOD
                else:
                    new_dp[next_j][next_xor] = dp[j][xor]
        
        # Add subsequence with just current number
        if x in new_dp[1]:
            new_dp[1][x] = (new_dp[1][x] + 1) % MOD
        else:
            new_dp[1][x] = 1
            
        dp = new_dp

    ans = 0
    # For each length that's multiple of M
    for j in range(M):
        if j == 0:
            continue
        if j % M == 0:
            # Add contribution of each XOR value
            for xor in dp[j]:
                ans = (ans + pow(xor, K, MOD) * dp[j][xor]) % MOD
    
    print(ans)

solve()