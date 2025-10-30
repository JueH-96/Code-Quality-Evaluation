def solve():
    n, a, x, y = map(int, input().split())
    
    dp = {}
    
    def get_cost(curr_n):
        if curr_n <= 0:
            return 0
        if curr_n in dp:
            return dp[curr_n]
        
        cost_a = x + get_cost(curr_n // a)
        
        cost_dice = y
        expected_cost_dice = 0
        for i in range(1, 7):
            expected_cost_dice += get_cost(curr_n // i) / 6
        cost_dice += expected_cost_dice
        
        dp[curr_n] = min(cost_a, cost_dice)
        return dp[curr_n]
    
    print(f"{get_cost(n):.10f}")

solve()