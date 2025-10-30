import sys
from functools import lru_cache

def solve():
    N, A, X, Y = map(int, input().split())
    
    # For large N, we need to find a threshold where we can compute exactly
    # and use approximation for larger values
    threshold = min(N, 100000)
    
    # Memoization for expected costs
    memo = {}
    
    def expected_cost(n):
        if n == 0:
            return 0.0
        
        if n in memo:
            return memo[n]
        
        # Option 1: Pay X to divide by A
        cost1 = X + expected_cost(n // A)
        
        # Option 2: Pay Y to roll die
        cost2 = Y
        die_expected = 0.0
        for b in range(1, 7):
            die_expected += expected_cost(n // b)
        cost2 += die_expected / 6.0
        
        result = min(cost1, cost2)
        memo[n] = result
        return result
    
    # For small values, compute exactly
    if N <= threshold:
        return expected_cost(N)
    
    # For large N, we need a different approach
    # The key insight is that for very large N, we can approximate
    # by looking at the pattern of smaller values
    
    # Compute expected costs for values near the threshold
    costs = []
    for i in range(max(1, threshold - 1000), threshold + 1):
        costs.append(expected_cost(i))
    
    # For large N, we can use the fact that the cost grows approximately
    # logarithmically. We'll simulate the process more directly.
    
    def solve_large(n):
        if n <= threshold:
            return expected_cost(n)
        
        # For large n, compute the expected cost by considering
        # what happens in the next step
        cost1 = X + solve_large(n // A)
        
        cost2 = Y
        die_expected = 0.0
        for b in range(1, 7):
            die_expected += solve_large(n // b)
        cost2 += die_expected / 6.0
        
        return min(cost1, cost2)
    
    # Use iterative approach to avoid recursion depth issues
    def solve_iterative(n):
        # Use BFS-like approach with memoization
        from collections import deque
        
        queue = deque([n])
        costs = {0: 0.0}
        
        # Process in reverse order (from small to large)
        to_process = []
        visited = set()
        
        def collect_states(curr):
            if curr in visited or curr == 0:
                return
            visited.add(curr)
            to_process.append(curr)
            
            # Add states we might transition to
            collect_states(curr // A)
            for b in range(1, 7):
                collect_states(curr // b)
        
        collect_states(n)
        to_process.sort()  # Process smaller states first
        
        for state in to_process:
            if state in costs:
                continue
                
            # Option 1: divide by A
            cost1 = X + costs.get(state // A, 0)
            
            # Option 2: roll die
            cost2 = Y
            for b in range(1, 7):
                cost2 += costs.get(state // b, 0) / 6.0
            
            costs[state] = min(cost1, cost2)
        
        return costs.get(n, 0)
    
    return solve_iterative(N)

result = solve()
print(f"{result:.15f}")