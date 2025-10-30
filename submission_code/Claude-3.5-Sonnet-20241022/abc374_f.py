def solve():
    # Read input
    N, K, X = map(int, input().split())
    T = list(map(int, input().split()))
    
    # dp[i][j] represents minimum dissatisfaction for orders[i:] 
    # when next available shipping day is j
    dp = {}
    
    def get_dp(pos, next_available):
        if pos >= N:  # All orders processed
            return 0
        
        if (pos, next_available) in dp:
            return dp[(pos, next_available)]
        
        # Find the earliest possible shipping day for current batch
        ship_day = max(next_available, T[pos])
        
        # Try all possible batch sizes (1 to K)
        min_dissatisfaction = float('inf')
        for batch_size in range(1, K + 1):
            if pos + batch_size > N:
                break
                
            # Calculate dissatisfaction for current batch
            current_dissatisfaction = 0
            for i in range(batch_size):
                current_dissatisfaction += ship_day - T[pos + i]
            
            # Recursively solve for remaining orders
            next_result = get_dp(pos + batch_size, ship_day + X)
            
            min_dissatisfaction = min(min_dissatisfaction, 
                                    current_dissatisfaction + next_result)
        
        dp[(pos, next_available)] = min_dissatisfaction
        return min_dissatisfaction
    
    # Get result starting from first order with day 1 as next available
    result = get_dp(0, 1)
    print(result)

solve()