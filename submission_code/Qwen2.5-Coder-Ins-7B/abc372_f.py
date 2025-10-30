MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = []
    for i in range(M):
        X = int(data[3 + 2 * i]) - 1
        Y = int(data[4 + 2 * i]) - 1
        edges.append((X, Y))
    
    # Initialize DP table
    dp = [[0] * N for _ in range(K + 1)]
    dp[0][0] = 1
    
    for k in range(K):
        for i in range(N):
            for x, y in edges:
                if i == x:
                    dp[k + 1][y] = (dp[k + 1][y] + dp[k][i]) % MOD
    
    result = sum(dp[K]) % MOD
    print(result)

if __name__ == "__main__":
    main()