MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    X = list(map(int, data[3:3+M]))
    
    # Initialize DP table
    dp = [[0] * (M+1) for _ in range(N+1)]
    dp[0][0] = 1
    
    for i in range(N):
        for j in range(M+1):
            if dp[i][j] == 0:
                continue
            for c in range(1, K+1):
                if c == X[j]:
                    if j+1 <= M:
                        dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
                else:
                    dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD
    
    T = sum(dp[N][j] for j in range(M)) % MOD
    
    print(T)

if __name__ == '__main__':
    main()