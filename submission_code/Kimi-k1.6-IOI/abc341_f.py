def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]  # 1-based
    for _ in range(M):
        u = int(data[idx])
        v = int(data[idx+1])
        adj[u].append(v)
        adj[v].append(u)
        idx +=2
    
    W = list(map(int, data[idx:idx+N]))
    idx +=N
    A = list(map(int, data[idx:idx+N]))
    
    # Create sorted list of nodes by their W value
    sorted_nodes = sorted([(W[i], i+1) for i in range(N)], key=lambda x: (x[0], x[1]))
    
    c = [0] * (N +1)  # 1-based
    
    for w, u in sorted_nodes:
        current_W = w
        capacity = current_W -1
        eligible = []
        # Collect eligible neighbors
        for v in adj[u]:
            if W[v-1] < current_W:
                eligible.append( (W[v-1], c[v]) )
        # Solve knapsack
        dp = [-float('inf')] * (capacity +1)
        if capacity >=0:
            dp[0] = 0
        for weight, val in eligible:
            for j in range(capacity, weight-1, -1):
                if dp[j - weight] + val > dp[j]:
                    dp[j] = dp[j - weight] + val
        max_sum = max(dp) if dp else 0
        c[u] = 1 + max_sum
    
    total = 0
    for i in range(1, N+1):
        total += A[i-1] * c[i]
    print(total)
    
if __name__ == '__main__':
    main()