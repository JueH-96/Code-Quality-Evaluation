# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    # Read N, M, K
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    K = int(data[index + 2])
    index += 3
    
    # Initialize adjacency list
    adj = [[] for _ in range(N + 1)]
    
    # Add the cycle edges
    for i in range(1, N + 1):
        adj[i].append(i % N + 1)
    
    # Add the additional M edges
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        index += 2
        adj[X].append(Y)
    
    # Initialize DP table
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    dp[0][1] = 1  # Start at vertex 1 with 0 moves
    
    # Fill the DP table
    for i in range(K):
        for v in range(1, N + 1):
            if dp[i][v] > 0:
                for u in adj[v]:
                    dp[i + 1][u] = (dp[i + 1][u] + dp[i][v]) % MOD
    
    # Calculate the result
    result = sum(dp[K][v] for v in range(1, N + 1)) % MOD
    print(result)

main()