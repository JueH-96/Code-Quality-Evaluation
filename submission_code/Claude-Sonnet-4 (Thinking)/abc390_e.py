def solve():
    N, X = map(int, input().split())
    foods = []
    for _ in range(N):
        v, a, c = map(int, input().split())
        foods.append((v, a, c))
    
    def can_achieve(target):
        if target == 0:
            return True
        
        MAX_V = 300  # Limit vitamin amounts to keep DP feasible
        
        # Separate foods by vitamin type
        vitamin_foods = [[] for _ in range(4)]
        for v, a, c in foods:
            vitamin_foods[v].append((a, c))
        
        # DP for vitamins 1 and 2: dp[v1][v2] = min calories to get v1 units of vitamin 1 and v2 units of vitamin 2
        dp = [[float('inf')] * (MAX_V + 1) for _ in range(MAX_V + 1)]
        dp[0][0] = 0
        
        # Process vitamin 1 foods
        for a, c in vitamin_foods[1]:
            new_dp = [row[:] for row in dp]
            for v1 in range(MAX_V + 1):
                for v2 in range(MAX_V + 1):
                    if dp[v1][v2] != float('inf') and v1 + a <= MAX_V:
                        new_dp[v1 + a][v2] = min(new_dp[v1 + a][v2], dp[v1][v2] + c)
            dp = new_dp
        
        # Process vitamin 2 foods
        for a, c in vitamin_foods[2]:
            new_dp = [row[:] for row in dp]
            for v1 in range(MAX_V + 1):
                for v2 in range(MAX_V + 1):
                    if dp[v1][v2] != float('inf') and v2 + a <= MAX_V:
                        new_dp[v1][v2 + a] = min(new_dp[v1][v2 + a], dp[v1][v2] + c)
            dp = new_dp
        
        # Check each (v1, v2) where both v1 >= target and v2 >= target
        for v1 in range(target, MAX_V + 1):
            for v2 in range(target, MAX_V + 1):
                if dp[v1][v2] == float('inf') or dp[v1][v2] > X:
                    continue
                
                remaining_calories = X - dp[v1][v2]
                
                # Knapsack for vitamin 3 with remaining calories
                dp3 = [0] * (remaining_calories + 1)
                for a, c in vitamin_foods[3]:
                    for j in range(remaining_calories, c - 1, -1):
                        dp3[j] = max(dp3[j], dp3[j - c] + a)
                
                if max(dp3) >= target:
                    return True
        
        return False
    
    # Binary search on the answer
    left, right = 0, 2000  # Upper bound estimate
    while left < right:
        mid = (left + right + 1) // 2
        if can_achieve(mid):
            left = mid
        else:
            right = mid - 1
    
    print(left)

solve()