def solve():
    n, a, x, y = map(int, input().split())
    
    dp = {}
    
    def calculate_cost(current_n):
        if current_n <= 0:
            return 0.0
        
        if current_n in dp:
            return dp[current_n]
        
        cost_a = x + calculate_cost(current_n // a)
        
        cost_dice = y
        expected_cost_dice = 0.0
        for i in range(1, 7):
            expected_cost_dice += calculate_cost(current_n // i) / 6.0
        cost_dice += expected_cost_dice
        
        dp[current_n] = min(cost_a, cost_dice)
        return dp[current_n]
    
    print(calculate_cost(n))

solve()