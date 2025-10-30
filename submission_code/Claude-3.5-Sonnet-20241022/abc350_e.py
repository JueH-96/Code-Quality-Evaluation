def solve(N, A, X, Y):
    dp = {}
    
    def expected_cost(n):
        if n == 0:
            return 0
        if n in dp:
            return dp[n]
            
        # Option 1: Use fixed division by A
        cost1 = X + expected_cost(n // A)
        
        # Option 2: Roll die
        cost2 = Y
        prob = 1/6
        exp = 0
        for roll in range(1, 7):
            next_n = n // roll
            exp += prob * expected_cost(next_n)
        cost2 += exp
        
        dp[n] = min(cost1, cost2)
        return dp[n]
        
    return expected_cost(N)

N, A, X, Y = map(int, input().split())
print(solve(N, A, X, Y))