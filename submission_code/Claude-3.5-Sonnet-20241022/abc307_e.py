def solve():
    MOD = 998244353
    N, M = map(int, input().split())
    
    # dp[i][j] represents number of valid ways for i people where last person has value j
    dp = [[0] * M for _ in range(2)]
    
    # Initialize first person - can take any value from 0 to M-1
    for j in range(M):
        dp[1][j] = 1
        
    curr = 1
    # For each person from 2 to N
    for i in range(2, N+1):
        curr ^= 1
        prev = curr^1
        
        # For each possible value of current person
        for j in range(M):
            dp[curr][j] = 0
            # Add contributions from all valid previous values
            for k in range(M):
                if j != k:
                    dp[curr][j] = (dp[curr][j] + dp[prev][k]) % MOD
                    
    # For last person, need to also check first person
    ans = 0
    # For each possible value of last person
    for j in range(M):
        # For each possible value of first person
        for k in range(M):
            if j != k:
                ans = (ans + dp[curr][j]) % MOD
                
    print(ans)

solve()