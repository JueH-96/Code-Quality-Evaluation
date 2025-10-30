def solve():
    N, X = map(int, input().split())
    teeth = []
    for _ in range(N):
        u, d = map(int, input().split())
        teeth.append((u, d))
    
    # All possible H values are sums of original teeth
    possible_h = set()
    for u, d in teeth:
        possible_h.add(u + d)
    
    min_cost = float('inf')
    
    for H in possible_h:
        # Check feasibility and get valid ranges
        ranges = []
        feasible = True
        
        for u, d in teeth:
            max_u = min(u, H - 1) if H > 1 else 0
            min_u = max(0, H - d)
            
            if min_u > max_u or H - max_u > d or H - min_u > d:
                feasible = False
                break
                
            ranges.append((min_u, max_u))
        
        if not feasible:
            continue
        
        # DP with coordinate compression
        prev_costs = {}
        
        # Initialize first tooth
        for u in range(ranges[0][0], ranges[0][1] + 1):
            cost = (teeth[0][0] - u) + (teeth[0][1] - (H - u))
            prev_costs[u] = cost
        
        # Process remaining teeth
        for i in range(1, N):
            curr_costs = {}
            min_u, max_u = ranges[i]
            
            for u in range(min_u, max_u + 1):
                cost_this = (teeth[i][0] - u) + (teeth[i][1] - (H - u))
                best = float('inf')
                
                for prev_u, prev_cost in prev_costs.items():
                    if abs(u - prev_u) <= X:
                        best = min(best, prev_cost + cost_this)
                
                if best < float('inf'):
                    curr_costs[u] = best
            
            prev_costs = curr_costs
        
        if prev_costs:
            min_cost = min(min_cost, min(prev_costs.values()))
    
    print(min_cost)

solve()