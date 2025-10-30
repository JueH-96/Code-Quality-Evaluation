# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = [[] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        edges[i].append(i % N + 1)
    
    index = 3
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        edges[X].append(Y)
        index += 2
    
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1
    
    for k in range(1, K + 1):
        for v in range(1, N + 1):
            for u in edges[v]:
                dp[k][u] = (dp[k][u] + dp[k - 1][v]) % MOD
    
    result = sum(dp[K][v] for v in range(1, N + 1)) % MOD
    print(result)

if __name__ == "__main__":
    main()