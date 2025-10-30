def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:]))
    
    # dp[i][k] will store the number of ways to fill the sequence up to index i
    # such that the maximum value used is k
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    # Base case: There's one way to have an empty sequence with max value 0
    dp[0][0] = 1
    
    for i in range(1, N + 1):
        if B[i - 1] == -1:
            # If B[i-1] is -1, we can choose any value from 1 to M
            for k in range(1, M + 1):
                for j in range(k, M + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][k - 1]) % MOD
        else:
            # If B[i-1] is a specific number, we can only use that number
            b = B[i - 1]
            for j in range(b, M + 1):
                dp[i][j] = (dp[i][j] + dp[i - 1][b - 1]) % MOD
    
    # Sum up all ways to fill the sequence with any maximum value
    result = sum(dp[N][k] for k in range(1, M + 1)) % MOD
    
    print(result)