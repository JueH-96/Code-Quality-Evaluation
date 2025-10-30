# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:]))
    
    # Count the number of -1 in B
    q = B.count(-1)
    
    # If there are no -1, we can directly compute f(B)
    if q == 0:
        print(compute_f(B))
        return
    
    # dp[i][j] will store the number of ways to form valid sequences up to position i with value j
    dp = [[0] * (M + 1) for _ in range(N)]
    
    # Initialize dp for the first position
    if B[0] == -1:
        for j in range(1, M + 1):
            dp[0][j] = 1
    else:
        dp[0][B[0]] = 1
    
    # Fill dp table
    for i in range(1, N):
        if B[i] == -1:
            for j in range(1, M + 1):
                for k in range(1, j + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD
        else:
            for k in range(1, B[i] + 1):
                dp[i][B[i]] = (dp[i][B[i]] + dp[i - 1][k]) % MOD
    
    # Sum up all valid sequences
    total_ways = sum(dp[N - 1]) % MOD
    
    # Compute the sum of f(B') over all possible B'
    result = (total_ways * compute_f(B)) % MOD
    print(result)

def compute_f(B):
    N = len(B)
    parent = list(range(N))
    rank = [0] * N
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
    
    for i in range(N):
        for j in range(i + 1, N):
            if B[i] <= B[j]:
                union(i, j)
    
    # Count the number of connected components
    components = len(set(find(i) for i in range(N)))
    return components