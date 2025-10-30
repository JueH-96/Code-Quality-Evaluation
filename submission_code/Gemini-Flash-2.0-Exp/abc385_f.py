def solve():
    n = int(input())
    buildings = []
    for _ in range(n):
        x, h = map(int, input().split())
        buildings.append((x, h))

    def can_see_all(height):
        for i in range(n):
            can_see = False
            for h_i in [0, buildings[i][1]]:
                can_see_from_h_i = True
                for j in range(n):
                    if i == j:
                        continue
                    
                    x_i, h_i_building = buildings[i]
                    x_j, h_j = buildings[j]
                    
                    # Check if the line segment intersects building j
                    
                    # Line equation: y = mx + c
                    m = (h_i - height) / x_i
                    c = height
                    
                    # Check if building j intersects the line segment
                    
                    # Check if the line segment intersects the bounding box of building j
                    if (x_j < x_i and x_j > 0) or (x_j > x_i and x_j < 0):
                        
                        # Check if the line segment intersects the building j
                        
                        # Check if the line segment intersects the top or bottom of building j
                        y_at_x_j = m * x_j + c
                        if 0 <= y_at_x_j <= h_j:
                            can_see_from_h_i = False
                            break
                if can_see_from_h_i:
                    can_see = True
                    break
            if not can_see:
                return False
        return True

    if can_see_all(0):
        print("-1")
        return

    low = 0
    high = 10**9
    
    for _ in range(100):
        mid = (low + high) / 2
        if can_see_all(mid):
            low = mid
        else:
            high = mid
            
    print(f"{high:.20f}")

solve()