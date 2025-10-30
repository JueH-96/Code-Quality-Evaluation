def solve():
    MOD = 998244353
    
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    
    # dp[i][j] represents number of sequences of length i 
    # where we have matched exactly j characters of sequence X
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    for i in range(N):
        for j in range(M + 1):
            if dp[i][j] == 0:
                continue
            
            # Try each possible character
            for c in range(1, K + 1):
                if j < M and c == X[j]:
                    # This character matches the next character in X
                    dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
                else:
                    # This character doesn't advance our match with X
                    dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
    
    # Total sequences that don't contain X as a subsequence
    result = sum(dp[N][j] for j in range(M)) % MOD
    
    print(result)

solve()