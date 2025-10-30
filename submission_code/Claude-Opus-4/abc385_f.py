def solve():
    n = int(input())
    buildings = []
    for _ in range(n):
        x, h = map(int, input().split())
        buildings.append((x, h))
    
    # For each building, find the minimum height from which it's visible
    max_blocking_height = 0.0
    
    for i in range(n):
        xi, hi = buildings[i]
        
        # Find the maximum slope from origin to any building before i
        max_slope = 0.0
        for j in range(i):
            xj, hj = buildings[j]
            slope = hj / xj
            max_slope = max(max_slope, slope)
        
        # If the line with max_slope passes below the top of building i, 
        # then building i is visible from height 0
        if max_slope * xi <= hi:
            continue
        
        # Otherwise, find the height h such that the line from (0, h) 
        # through the highest blocking point just touches the top of building i
        # The highest blocking point is at (xk, hk) where k gives max slope
        # We need: (hi - h) / xi = (hk - h) / xk
        # This gives: h = (hi * xk - hk * xi) / (xk - xi)
        
        for j in range(i):
            xj, hj = buildings[j]
            if hj / xj == max_slope:
                # This building j creates the maximum slope
                # Calculate the height needed to just see building i over building j
                h_needed = (hi * xj - hj * xi) / (xj - xi)
                max_blocking_height = max(max_blocking_height, h_needed)
                break
    
    # Check if all buildings are visible from height 0
    all_visible_from_zero = True
    for i in range(n):
        xi, hi = buildings[i]
        blocked = False
        for j in range(i):
            xj, hj = buildings[j]
            # Check if building j blocks building i when viewed from (0, 0)
            if hj / xj > hi / xi:
                blocked = True
                break
        if blocked:
            all_visible_from_zero = False
            break
    
    if all_visible_from_zero:
        print(-1)
    else:
        print(f"{max_blocking_height:.18f}")

solve()