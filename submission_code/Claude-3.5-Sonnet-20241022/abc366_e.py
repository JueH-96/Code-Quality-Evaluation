def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def solve():
    N, D = map(int, input().split())
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))
    
    # Find bounding box of points
    min_x = min(x for x,y in points)
    max_x = max(x for x,y in points)
    min_y = min(y for x,y in points)
    max_y = max(y for x,y in points)
    
    # Extend search space by D in each direction
    min_x = min_x - D
    max_x = max_x + D
    min_y = min_y - D
    max_y = max_y + D
    
    count = 0
    # For each point in extended bounding box
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            total_dist = 0
            # Calculate sum of manhattan distances to all input points
            for px, py in points:
                total_dist += manhattan_distance(x, y, px, py)
                if total_dist > D:
                    break
            if total_dist <= D:
                count += 1
                
    print(count)

solve()