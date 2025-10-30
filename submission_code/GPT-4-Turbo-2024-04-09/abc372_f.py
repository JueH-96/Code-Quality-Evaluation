def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    edges = [[] for _ in range(N + 1)]
    
    # Add the cyclic edges
    for i in range(1, N):
        edges[i].append(i + 1)
    edges[N].append(1)
    
    # Add the additional M edges
    index = 3
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        edges[X].append(Y)
        index += 2
    
    # dp[k][v] represents the number of ways to reach vertex v in exactly k steps
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1  # Start at vertex 1
    
    for k in range(K):
        for v in range(1, N + 1):
            if dp[k][v] > 0:
                for to in edges[v]:
                    dp[k + 1][to] = (dp[k + 1][to] + dp[k][v]) % MOD
    
    # Sum all ways to reach any vertex in exactly K steps
    result = 0
    for v in range(1, N + 1):
        result = (result + dp[K][v]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()