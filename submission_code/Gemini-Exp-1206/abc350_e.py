def solve():
    n, a, x, y = map(int, input().split())
    
    dp = {}

    def get_dp(val):
        if val == 0:
            return 0
        if val in dp:
            return dp[val]
        
        res = float('inf')
        
        res = min(res, x + get_dp(val // a))
        
        expected_cost = 0
        for i in range(1, 7):
            expected_cost += get_dp(val // i)
        expected_cost /= 6
        res = min(res, y + expected_cost)
        
        dp[val] = res
        return res

    print(get_dp(n))

solve()