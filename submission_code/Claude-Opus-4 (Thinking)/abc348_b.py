# Read number of points
n = int(input())

# Read coordinates
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# For each point, find the farthest point
for i in range(n):
    max_dist = -1
    farthest_id = -1
    
    for j in range(n):
        if i == j:
            continue
        
        # Calculate squared distance (no need for sqrt since we're just comparing)
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        dist = dx * dx + dy * dy
        
        # Update if this is farther, or same distance but smaller ID
        if dist > max_dist:
            max_dist = dist
            farthest_id = j + 1  # IDs are 1-indexed
        elif dist == max_dist and j + 1 < farthest_id:
            farthest_id = j + 1
    
    print(farthest_id)