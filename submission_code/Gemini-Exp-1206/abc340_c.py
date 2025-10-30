def solve():
    n = int(input())
    
    def calculate_cost(x):
        if x < 2:
            return 0
        
        dp = {}
        
        def recurse(val):
            if val < 2:
                return 0
            if val in dp:
                return dp[val]
            
            cost = val + recurse(val // 2) + recurse((val + 1) // 2)
            dp[val] = cost
            return cost
        
        return recurse(x)

    print(calculate_cost(n))

solve()