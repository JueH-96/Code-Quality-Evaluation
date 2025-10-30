def keyence_delivery(N, K, X, T):
    memo = {}
    
    def dp(i, j):
        """
        Minimum dissatisfaction to ship all remaining orders, where:
        i: index of the first unshipped order
        j: earliest day we can make the next shipment
        """
        if i == N:
            return 0
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        # We need to decide how many consecutive orders to ship in the next batch
        min_dissatisfaction = float('inf')
        for k in range(1, min(K, N-i) + 1):
            # Earliest day to ship this batch
            ship_day = max(j, T[i+k-1])
            
            # Dissatisfaction for this batch
            batch_dissatisfaction = k * ship_day - sum(T[i:i+k])
            
            # Recurrence
            total_dissatisfaction = batch_dissatisfaction + dp(i+k, ship_day + X)
            
            min_dissatisfaction = min(min_dissatisfaction, total_dissatisfaction)
        
        memo[(i, j)] = min_dissatisfaction
        return min_dissatisfaction
    
    return dp(0, 0)

# Reading input and solving the problem
N, K, X = map(int, input().split())
T = list(map(int, input().split()))
print(keyence_delivery(N, K, X, T))