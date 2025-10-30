# Read the input
N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

# For each point, find the farthest point
for i in range(N):
    max_sq_distance = -1
    farthest_point_id = -1
    
    for j in range(N):
        if i == j:
            continue  # Skip the same point
        
        # Calculate squared Euclidean distance (avoid square root for precision)
        sq_distance = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
        
        if sq_distance > max_sq_distance:
            max_sq_distance = sq_distance
            farthest_point_id = j+1  # +1 because the ID starts from 1
        elif sq_distance == max_sq_distance and j+1 < farthest_point_id:
            farthest_point_id = j+1
    
    print(farthest_point_id)