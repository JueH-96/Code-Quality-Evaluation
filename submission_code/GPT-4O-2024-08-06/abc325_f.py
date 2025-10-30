def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    D = list(map(int, data[idx:idx + N]))
    idx += N
    
    L1, C1, K1 = map(int, data[idx:idx + 3])
    idx += 3
    
    L2, C2, K2 = map(int, data[idx:idx + 3])
    idx += 3
    
    INF = float('inf')
    
    # Function to calculate minimum cost to cover a single section
    def min_cost_to_cover(length):
        # dp[x] = minimum cost to cover length x
        dp = [INF] * (length + 1)
        dp[0] = 0
        
        # Try using type-1 sensors
        for k in range(1, K1 + 1):
            for x in range(length, L1 - 1, -1):
                dp[x] = min(dp[x], dp[x - L1] + C1)
        
        # Try using type-2 sensors
        for k in range(1, K2 + 1):
            for x in range(length, L2 - 1, -1):
                dp[x] = min(dp[x], dp[x - L2] + C2)
        
        return dp[length]
    
    total_cost = 0
    
    for length in D:
        cost = min_cost_to_cover(length)
        if cost == INF:
            print(-1)
            return
        total_cost += cost
    
    print(total_cost)