# Read N
N = int(input())

# Read all points. Store them in a list.
# points[k] will store coordinates for point with ID (k+1)
points = []
for _ in range(N):
    x_coord, y_coord = map(int, input().split())
    points.append((x_coord, y_coord))

# For each point i (0-indexed in the list 'points')
for i in range(N):
    # (x1, y1) are coordinates of the current point P_{i+1}
    x1, y1 = points[i]
    
    # Initialize variables to keep track of the farthest point found so far for P_{i+1}
    # max_dist_sq stores the squared Euclidean distance to that farthest point
    # farthest_point_id stores the 1-based ID of that point
    max_dist_sq = -1  # Squared distances are non-negative, so -1 is a safe initial value
    farthest_point_id = -1 # Will hold the 1-based ID
    
    # Iterate over all other points j (0-indexed in 'points') to find the one farthest from point i
    for j in range(N):
        # A point cannot be the farthest from itself
        if i == j:
            continue
        
        # (x2, y2) are coordinates of a candidate point P_{j+1}
        # Its ID is j+1
        x2, y2 = points[j]
        candidate_id = j + 1 
        
        # Calculate squared Euclidean distance
        dx = x1 - x2
        dy = y1 - y2
        current_dist_sq = dx*dx + dy*dy
        
        # Check if this point is farther than the current farthest
        if current_dist_sq > max_dist_sq:
            max_dist_sq = current_dist_sq
            farthest_point_id = candidate_id
        elif current_dist_sq == max_dist_sq:
            # If distances are equal, choose the point with the smaller ID
            # Since farthest_point_id is initialized to -1, this check also correctly
            # handles the very first point if its distance sets the initial max_dist_sq,
            # but practically, the first point j != i will fall into the ">" case.
            # Thus, when this "==" case is met, farthest_point_id will be a valid ID.
            if candidate_id < farthest_point_id:
                farthest_point_id = candidate_id
                # max_dist_sq remains the same
                
    # Print the ID of the farthest point found for point P_{i+1}
    print(farthest_point_id)