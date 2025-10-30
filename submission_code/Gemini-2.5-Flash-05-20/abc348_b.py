import sys

def solve():
    N = int(sys.stdin.readline())
    
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
        
    results = []
    
    # Iterate through each point i to find its farthest point
    for i in range(N):
        current_point = points[i]
        
        # Initialize max_dist_sq to a value smaller than any possible squared distance (min is 0)
        max_dist_sq = -1 
        # Initialize farthest_point_id to an invalid ID or a value that will be updated by the first candidate
        farthest_point_id = -1 
        
        # Iterate through all other points j to find the one farthest from current_point
        for j in range(N):
            if i == j:
                continue # A point cannot be farthest from itself
            
            other_point = points[j]
            
            # Calculate squared Euclidean distance to avoid sqrt and float issues
            dx = current_point[0] - other_point[0]
            dy = current_point[1] - other_point[1]
            dist_sq = dx*dx + dy*dy
            
            # Check if this point is farther
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                farthest_point_id = j + 1 # Store 1-indexed ID
            # If equally far, check if its ID is smaller
            elif dist_sq == max_dist_sq:
                if (j + 1) < farthest_point_id:
                    farthest_point_id = j + 1
        
        results.append(farthest_point_id)
        
    # Print results for each point
    for res_id in results:
        sys.stdout.write(str(res_id) + "
")

solve()