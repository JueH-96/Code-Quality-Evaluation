import sys
from math import ceil

def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    
    # For each section, find all possible ways to cover it and their costs
    section_options = []
    
    for d in D:
        options = []
        
        # Try all possible combinations of type-1 sensors (from 0 to enough to cover d)
        max_type1_needed = ceil(d / L1) if L1 > 0 else K1 + 1
        
        for x in range(min(K1 + 1, max_type1_needed + 1)):
            # Calculate minimum type-2 sensors needed
            remaining = max(0, d - x * L1)
            if remaining == 0:
                y = 0
            else:
                y = ceil(remaining / L2) if L2 > 0 else float('inf')
            
            if y <= K2 and y != float('inf'):
                cost = x * C1 + y * C2
                options.append((x, y, cost))
        
        if not options:
            print(-1)
            return
            
        section_options.append(options)
    
    # Now we need to find the minimum cost assignment that respects sensor limits
    # This is a complex optimization problem, so we'll use dynamic programming
    
    # State: (section_index, type1_used, type2_used) -> min_cost
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(section_idx, type1_used, type2_used):
        if section_idx == N:
            return 0
        
        min_cost = float('inf')
        
        for x, y, cost in section_options[section_idx]:
            if type1_used + x <= K1 and type2_used + y <= K2:
                remaining_cost = dp(section_idx + 1, type1_used + x, type2_used + y)
                if remaining_cost != float('inf'):
                    min_cost = min(min_cost, cost + remaining_cost)
        
        return min_cost
    
    result = dp(0, 0, 0)
    
    if result == float('inf'):
        print(-1)
    else:
        print(result)

solve()