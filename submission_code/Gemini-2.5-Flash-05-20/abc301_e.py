import collections
import sys

def solve():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    start_pos = None
    goal_pos = None
    candy_pos = []
    
    # Store all important points and their original grid coordinates
    # S -> index 0
    # Candies -> indices 1 to K
    # G -> index K+1
    all_points_coords = []
    
    # Find S, G, and all 'o' positions
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
            elif grid[r][c] == 'o':
                candy_pos.append((r, c))

    # Build the list of important points in order: S, o's, G
    # This ordering simplifies mask calculations later (candy_idx - 1)
    all_points_coords.append(start_pos)
    all_points_coords.extend(candy_pos)
    all_points_coords.append(goal_pos)

    num_candies = len(candy_pos)
    num_total_points = len(all_points_coords) # K + 2

    S_idx = 0
    G_idx = num_total_points - 1 # K + 1

    # Precompute shortest distances between all pairs of important points using BFS
    dist_matrix = [[float('inf')] * num_total_points for _ in range(num_total_points)]

    for i in range(num_total_points):
        start_r, start_c = all_points_coords[i]
        
        q = collections.deque([(start_r, start_c)]) # Stores (r, c)
        cell_distances = [[-1] * W for _ in range(H)] # -1 means not visited
        cell_distances[start_r][start_c] = 0

        while q:
            r, c = q.popleft() # O(1)
            d = cell_distances[r][c]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                # Check bounds, wall, and visited status
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and cell_distances[nr][nc] == -1:
                    cell_distances[nr][nc] = d + 1
                    q.append((nr, nc))
        
        # Populate dist_matrix for paths from point i to all other points j
        for j in range(num_total_points):
            target_r, target_c = all_points_coords[j]
            if cell_distances[target_r][target_c] != -1:
                dist_matrix[i][j] = cell_distances[target_r][target_c]

    # Dynamic Programming with Bitmask
    # dp[mask][last_idx] = minimum time to reach all_points_coords[last_idx] having visited
    #                       exactly the candy squares specified by 'mask'.
    # mask: a bitmask where the j-th bit (0-indexed) is set if the j-th candy (all_points_coords[j+1]) is visited.
    dp = [[float('inf')] * num_total_points for _ in range(1 << num_candies)]

    # Initial state: starting at S (index 0), no candies visited, time 0
    dp[0][S_idx] = 0

    # Iterate through masks in increasing order to ensure dependencies are met
    for mask in range(1 << num_candies):
        for u_idx in range(num_total_points): # Current point (from which we transition)
            if dp[mask][u_idx] == float('inf'):
                continue

            # Try to move to any other important point v_idx
            for v_idx in range(num_total_points): # Next point (to which we transition)
                if u_idx == v_idx: # No self-loops needed; min path from point to itself is 0, handled by initial dist_matrix setup if needed
                    continue

                travel_cost = dist_matrix[u_idx][v_idx]
                if travel_cost == float('inf'): # Unreachable from u_idx to v_idx
                    continue

                new_time = dp[mask][u_idx] + travel_cost
                if new_time > T: # Path exceeds time limit
                    continue

                next_mask = mask
                # If v_idx is a candy square (indices 1 to num_candies)
                if 1 <= v_idx <= num_candies:
                    # Set the corresponding bit for this candy in the new mask
                    next_mask |= (1 << (v_idx - 1))
                
                # Update DP table if a shorter path to (next_mask, v_idx) is found
                if new_time < dp[next_mask][v_idx]:
                    dp[next_mask][v_idx] = new_time

    # Find the maximum number of candies visited that ends at G within time T
    max_candies_visited = -1 # Default to -1 if no path to G is possible within T

    for mask in range(1 << num_candies):
        if dp[mask][G_idx] <= T:
            # Count set bits in the mask to find number of candies visited
            candies_count = bin(mask).count('1')
            max_candies_visited = max(max_candies_visited, candies_count)

    sys.stdout.write(str(max_candies_visited) + "
")

# Call the solve function to run the program
solve()