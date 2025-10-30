# YOUR CODE HERE
def solve():
    n, x = map(int, input().split())
    teeth = []
    for _ in range(n):
        u, d = map(int, input().split())
        teeth.append((u, d))
    
    # Find the range of possible H values
    min_h = 2
    max_h = min(u + d for u, d in teeth)
    
    min_cost = float('inf')
    
    # Try each possible H
    for h in range(min_h, max_h + 1):
        # For this H, use DP to find minimum cost
        # dp[i][u] = min cost for teeth 0..i where tooth i has upper length u
        
        # For position 0
        dp_prev = {}
        u0, d0 = teeth[0]
        for u in range(max(0, h - d0), min(u0, h) + 1):
            d = h - u
            if d >= 0 and d <= d0:
                cost = (u0 - u) + (d0 - d)
                dp_prev[u] = cost
        
        # For positions 1 to n-1
        for i in range(1, n):
            dp_curr = {}
            ui, di = teeth[i]
            
            for u in range(max(0, h - di), min(ui, h) + 1):
                d = h - u
                if d >= 0 and d <= di:
                    cost = (ui - u) + (di - d)
                    
                    # Check all valid previous states
                    min_prev_cost = float('inf')
                    for prev_u, prev_cost in dp_prev.items():
                        if abs(u - prev_u) <= x:
                            min_prev_cost = min(min_prev_cost, prev_cost)
                    
                    if min_prev_cost != float('inf'):
                        dp_curr[u] = min_prev_cost + cost
            
            dp_prev = dp_curr
        
        # Find minimum cost for this H
        if dp_prev:
            min_cost = min(min_cost, min(dp_prev.values()))
    
    print(min_cost)

solve()