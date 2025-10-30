def solve():
    N, A, X, Y = map(int, input().split())
    
    memo = {}
    
    def expected_cost(n):
        if n == 0:
            return 0
        
        if n in memo:
            return memo[n]
        
        # Option 1: Use operation 1 (deterministic)
        cost1 = X + expected_cost(n // A)
        
        # Option 2: Use operation 2 (roll die)
        # Expected cost = Y + (1/6) * sum of expected costs for each outcome
        cost2 = Y
        for i in range(1, 7):
            cost2 += expected_cost(n // i) / 6
        
        # Choose the minimum
        result = min(cost1, cost2)
        memo[n] = result
        return result
    
    answer = expected_cost(N)
    print(f"{answer:.15f}")

solve()