def can_see_all_buildings(h0, buildings):
    # Check if we can see all buildings from point (0, h0)
    n = len(buildings)
    
    # For each building, check if we can see at least one point on it
    for i in range(n):
        x, h = buildings[i]
        can_see = False
        
        # Try to see the top of the building
        # Check if line of sight is blocked by any building in between
        blocked = False
        for j in range(n):
            if j == i:
                continue
            xj, hj = buildings[j]
            
            # Only check buildings between point (0,h0) and current building
            if 0 <= xj <= x:
                # Calculate height of line of sight at position xj
                # Using similar triangles
                h_line = h0 + (h - h0) * xj / x
                
                if h_line <= hj:
                    blocked = True
                    break
        
        if not blocked:
            can_see = True
            
        if not can_see:
            return False
            
    return True

def binary_search(buildings):
    left = 0.0
    right = 10**12  # Some large value
    
    # If we can see all buildings from height 0, return -1
    if can_see_all_buildings(0, buildings):
        return -1
        
    # Binary search for maximum height where we cannot see all buildings
    for _ in range(100):  # Sufficient iterations for required precision
        mid = (left + right) / 2
        if can_see_all_buildings(mid, buildings):
            right = mid
        else:
            left = mid
            
    return left

def main():
    # Read input
    N = int(input())
    buildings = []
    for _ in range(N):
        x, h = map(int, input().split())
        buildings.append((x, h))
    
    # Sort buildings by x-coordinate (though input is already sorted)
    buildings.sort()
    
    # Find and print result
    result = binary_search(buildings)
    print(result)

if __name__ == "__main__":
    main()