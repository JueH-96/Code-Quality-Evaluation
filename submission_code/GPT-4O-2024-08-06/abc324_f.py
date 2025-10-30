# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        b = int(data[index + 2])
        c = int(data[index + 3])
        edges.append((u, v, b, c))
        index += 4
    
    def can_achieve_ratio(r):
        # dp[v] will store the maximum value of transformed path sum to vertex v
        dp = [-float('inf')] * N
        dp[0] = 0  # Starting at vertex 1 (index 0)
        
        for u, v, b, c in edges:
            if dp[u] != -float('inf'):
                dp[v] = max(dp[v], dp[u] + b - r * c)
        
        return dp[N-1] >= 0
    
    # Binary search for the maximum ratio
    low, high = 0.0, 10000.0  # Since b_i, c_i <= 10000
    epsilon = 1e-9
    
    while high - low > epsilon:
        mid = (low + high) / 2.0
        if can_achieve_ratio(mid):
            low = mid
        else:
            high = mid
    
    print(f"{low:.15f}")