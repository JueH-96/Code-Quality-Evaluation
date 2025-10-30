# YOUR CODE HERE
import collections
import sys

# Function to perform Breadth-First Search (BFS) from a starting point (sr, sc)
# This BFS computes shortest path distances from (sr, sc) to all points specified in points_coords_set.
# It returns a dictionary mapping target coordinates to their shortest distance from (sr, sc).
# If a point is unreachable, its distance is float('inf').
def bfs(H, W, grid, start_r, start_c, points_coords_set):
    # The queue stores tuples of (row, column, distance_from_start)
    q = collections.deque([(start_r, start_c, 0)])
    
    # 'dists' dictionary stores the shortest distance found so far to each cell (r, c).
    # It also implicitly keeps track of visited cells.
    dists = { (start_r, start_c): 0 }
    
    # 'results' dictionary will store the shortest distances to the specific points of interest.
    # Initialize distances to infinity.
    results = {coord: float('inf') for coord in points_coords_set}
    
    # If the starting cell itself is a point of interest, record its distance (0).
    if (start_r, start_c) in results:
         results[(start_r, start_c)] = 0
    
    # Counter for how many points of interest have been reached.
    found_count = 0
    if (start_r, start_c) in points_coords_set:
         found_count = 1
    
    num_targets = len(points_coords_set)

    # Main BFS loop
    while q:
        r, c, d = q.popleft()

        # Optimization: If all target points have been found, we can stop the BFS early.
        # This is safe because BFS explores layer by layer, guaranteeing shortest paths are found first.
        if found_count == num_targets:
             break

        # Explore neighbors: up, down, left, right
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Check if the neighbor is within grid bounds, is not a wall ('#'), and has not been visited yet.
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and (nr, nc) not in dists:
                # Record the distance to this neighbor and mark it as visited.
                dists[(nr, nc)] = d + 1
                # Add the neighbor to the queue for further exploration.
                q.append((nr, nc, d + 1))
                
                # Check if this newly reached neighbor is one of the points of interest.
                if (nr, nc) in points_coords_set:
                    # If this is the first time reaching this point of interest, record its shortest distance.
                    if results[(nr, nc)] == float('inf'): 
                         results[(nr, nc)] = d + 1
                         found_count += 1
                         # Check again if all targets found after update; if so, can break inner loop
                         if found_count == num_targets:
                             break # Optimization: break from exploring neighbors loop
                                
    # Return the dictionary containing shortest distances to points of interest.
    return results


def solve():
    # Read input dimensions and time limit T using faster I/O
    H, W, T = map(int, sys.stdin.readline().split())
    # Read the grid layout
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    start_pos = None
    goal_pos = None
    candy_coords = []

    # Locate the starting point 'S', goal point 'G', and all candy locations 'o'.
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
            elif grid[r][c] == 'o':
                candy_coords.append((r, c))

    # Number of candy squares
    K = len(candy_coords)
    
    # Handle the trivial case where there are no candies.
    if K == 0:
        # Calculate the shortest distance directly from Start to Goal using BFS.
        bfs_result_s_to_g = bfs(H, W, grid, start_pos[0], start_pos[1], {goal_pos})
        dist_s_g = bfs_result_s_to_g.get(goal_pos, float('inf'))
        # If the path exists and is within the time limit T, print 0 candies visited.
        if dist_s_g <= T:
            print(0)
        # Otherwise, it's impossible to reach the goal within T moves.
        else:
            print("-1")
        return

    # Create a list of all points of interest: K candies, followed by Start, then Goal.
    # Assign indices: candies 0..K-1, Start K, Goal K+1.
    points = candy_coords + [start_pos, goal_pos]
    num_points = K + 2
    
    # Use a set of coordinates for points of interest for efficient lookup during BFS.
    points_coords_set = set(points)

    # Initialize a distance matrix to store shortest path distances between all pairs of points of interest.
    dist_matrix = [[float('inf')] * num_points for _ in range(num_points)]

    # Compute all-pairs shortest paths by running BFS starting from each point of interest.
    for i in range(num_points):
        start_r, start_c = points[i]
        
        # Run BFS from points[i] to find distances to all other points in points_coords_set.
        bfs_results = bfs(H, W, grid, start_r, start_c, points_coords_set) 
        
        # Populate the i-th row of the distance matrix using the BFS results.
        for j in range(num_points):
            target_coord = points[j]
            # Use .get() with default value float('inf') for safety if a point wasn't reached.
            dist_matrix[i][j] = bfs_results.get(target_coord, float('inf'))


    # Indices for Start (S) and Goal (G) in the points list and distance matrix.
    start_idx = K
    goal_idx = K + 1
    
    # Get the shortest distance directly from Start to Goal.
    dist_S_G = dist_matrix[start_idx][goal_idx]
    # First, check if it's possible to reach the Goal from Start within T moves *at all*.
    # If the distance is infinite (unreachable) or greater than T, print -1 and exit.
    if dist_S_G > T: 
        print("-1")
        return

    # Dynamic Programming part using bitmask.
    # State definition: dp[mask][i] = minimum distance required to start at S, 
    # visit exactly the set of candies represented by 'mask', and end at candy 'i'.
    # 'mask' is an integer where the k-th bit is set if the k-th candy (points[k]) has been visited.
    # 'i' is the index of the last visited candy (0 <= i < K).
    dp = [[float('inf')] * K for _ in range(1 << K)]

    # Base cases: Paths from Start (S) directly to each candy 'i'.
    for i in range(K):
        # Get the shortest distance from S to candy i.
        dist_s_to_i = dist_matrix[start_idx][i]
        # A path is valid only if candy 'i' is reachable from S and the path length is within T.
        if dist_s_to_i <= T:
             # Set the initial DP state for visiting only candy 'i'. Mask is (1 << i).
             dp[1 << i][i] = dist_s_to_i

    # Fill the DP table using transitions. Iterate through masks from 1 up to (2^K - 1).
    for mask in range(1, 1 << K):
        # For each mask, iterate through all possible last visited candies 'i'.
        for i in range(K):
            # Check if candy 'i' is actually included in the current set 'mask'.
            if (mask >> i) & 1:
                # If the current state (mask, i) is unreachable within T (distance is infinity), skip it.
                if dp[mask][i] == float('inf'): continue 

                current_dist = dp[mask][i] # Minimum distance found so far for state (mask, i).

                # Try extending the path from candy 'i' to another currently unvisited candy 'j'.
                for j in range(K):
                     # Check if candy 'j' is NOT included in the current set 'mask'.
                     if not ((mask >> j) & 1):
                        # Get the shortest distance between candy 'i' and candy 'j'.
                        dist_i_to_j = dist_matrix[i][j]
                        
                        # Check if candy 'j' is reachable from candy 'i'.
                        if dist_i_to_j != float('inf'):
                           # Calculate the total distance of the new path: S -> ... -> i -> j
                           new_dist = current_dist + dist_i_to_j
                           
                           # Update the DP state for the new path only if its total distance is within T.
                           if new_dist <= T: 
                                # The new mask includes candy 'j'.
                                new_mask = mask | (1 << j)
                                # Update the DP table for state (new_mask, j) with the minimum distance found.
                                dp[new_mask][j] = min(dp[new_mask][j], new_dist)

    # After filling the DP table, find the maximum number of candies that can be visited.
    # Initialize max_candies to 0, because we already established that reaching G from S directly 
    # (visiting 0 candies) is possible within T moves.
    max_candies = 0 

    # Iterate through all possible subsets of visited candies (all masks).
    # We only need to check masks > 0, since max_candies is already 0.
    for mask in range(1, 1 << K): 
        # Count the number of candies visited in the current subset 'mask'.
        num_candies_visited = bin(mask).count('1')
        
        # Consider paths ending at each candy 'i' within the current subset 'mask'.
        for i in range(K):
            # Check if candy 'i' is part of the current mask.
            if (mask >> i) & 1: 
                # Check if the state (mask, i) is reachable within T moves.
                if dp[mask][i] != float('inf'):
                    # Get the shortest distance from the last visited candy 'i' to the Goal (G).
                    dist_i_to_g = dist_matrix[i][goal_idx]
                    
                    # Check if Goal (G) is reachable from candy 'i'.
                    if dist_i_to_g != float('inf'):
                        # Calculate the total path distance: S -> ... -> i -> G.
                        total_dist = dp[mask][i] + dist_i_to_g
                        
                        # If this complete path's distance is within the time limit T.
                        if total_dist <= T:
                            # Update max_candies if the current path visits more candies than the current maximum.
                            max_candies = max(max_candies, num_candies_visited)

    # Print the final result: the maximum number of distinct candy squares visited.
    print(max_candies)

# Execute the main solution logic.
solve()