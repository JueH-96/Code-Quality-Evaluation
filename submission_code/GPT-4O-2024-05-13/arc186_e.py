# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def solve():
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    X = list(map(int, data[3:]))
    
    # Initialize dp table
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    # Fill dp table
    for i in range(N):
        for j in range(M + 1):
            if dp[i][j] == 0:
                continue
            for c in range(1, K + 1):
                if j < M and c == X[j]:
                    dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
                else:
                    dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
    
    # Sum up all valid sequences of length N that do not contain the full sequence X
    result = sum(dp[N][j] for j in range(M)) % MOD
    print(result)