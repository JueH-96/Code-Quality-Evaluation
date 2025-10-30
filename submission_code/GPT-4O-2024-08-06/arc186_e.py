# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    X = list(map(int, data[3:]))
    
    MOD = 998244353
    
    # dp[i][j] means the number of ways to construct a sequence of length i
    # such that the longest prefix of X that can be a subsequence is of length j.
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 1
    
    for i in range(N):
        for j in range(M + 1):
            if dp[i][j] == 0:
                continue
            for a in range(1, K + 1):
                if j < M and a == X[j]:
                    dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
    
    # We want the number of sequences of length N where the longest prefix of X
    # that can be a subsequence is less than M.
    result = sum(dp[N][j] for j in range(M)) % MOD
    print(result)

main()