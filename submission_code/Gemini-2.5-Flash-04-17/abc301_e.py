import sys
from collections import deque

# Function to read grid and find key locations
def parse_grid(H, W, grid_lines):
    grid = [list(line) for line in grid_lines]
    start_coords = None
    goal_coords = None
    candy_coords = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start_coords = (r, c)
            elif grid[r][c] == 'G':
                goal_coords = (r, c)
            elif grid[r][c] == 'o':
                candy_coords.append((r, c))
    return grid, start_coords, goal_coords, candy_coords

# BFS to find shortest distances from a starting point
def bfs(start_coords, grid, H, W):
    dist = [[float('inf')] * W for _ in range(H)]
    q = deque([(start_coords[0], start_coords[1], 0)])
    dist[start_coords[0]][start_coords[1]] = 0

    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    while q:
        r, c, d = q.popleft()

        # If we found a shorter path already, skip (redundant check, but safe)
        # if d > dist[r][c]:
        #     continue

        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                if dist[nr][nc] > d + 1:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc, d + 1))
    return dist

# Main logic
def solve():
    H, W, T = map(int, sys.stdin.readline().split())
    grid_lines = [sys.stdin.readline().strip() for _ in range(H)]

    grid, start_coords, goal_coords, candy_coords = parse_grid(H, W, grid_lines)

    # Key locations: S (0), Candies (1 to k), G (k+1)
    key_locations = [start_coords] + candy_coords + [goal_coords]
    k = len(candy_coords) # Number of candies
    N = len(key_locations) # Total number of key locations (k + 2)

    # Precompute all-pairs shortest paths between key locations
    # dists[i][j] = shortest distance from key_locations[i] to key_locations[j]
    dists = [[float('inf')] * N for _ in range(N)]
    dist_grids = [None] * N
    for i in range(N):
        # Run BFS starting from each key location
        dist_grids[i] = bfs(key_locations[i], grid, H, W)

    # Populate dists table using BFS results
    for i in range(N):
        for j in range(N):
            r, c = key_locations[j]
            dists[i][j] = dist_grids[i][r][c]

    # Precompute candies on *a* shortest path between key locations
    # segment_candies_mask[i][j] = bitmask of candies (original indices 0 to k-1)
    # that lie on a shortest path from key_locations[i] to key_locations[j]
    segment_candies_mask = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if dists[i][j] == float('inf'):
                continue # Cannot move from i to j

            mask = 0
            # Check each candy if it lies on a shortest path from i to j
            # m is the original index of the candy (0 to k-1)
            # candy_key_idx is its index in key_locations (1 to k)
            for m in range(k):
                candy_key_idx = m + 1
                
                # A candy C_m is on *a* shortest path from key_i to key_j
                # if dist(key_i, C_m) + dist(C_m, key_j) == dist(key_i, key_j)
                # provided the intermediate distances are finite.
                if dists[i][candy_key_idx] != float('inf') and dists[candy_key_idx][j] != float('inf') and \
                   dists[i][candy_key_idx] + dists[candy_key_idx][j] == dists[i][j]:
                   mask |= (1 << m)
            segment_candies_mask[i][j] = mask

    # DP state: dp[mask][current_key_idx] = minimum distance to reach current_key_idx,
    # having visited *at least* the set of candies represented by 'mask'.
    # mask: k bits. i-th bit (0 to k-1) is 1 if candy with original index i has been visited.
    # current_key_idx: index in key_locations (0 to N-1).
    dp = [[float('inf')] * N for _ in range(1 << k)]

    # Initial state: Start at S (key_idx 0), mask 0 (no candies visited yet), dist 0
    dp[0][0] = 0

    # DP transitions
    # Iterate through masks in increasing order. This ensures that when we process
    # a mask, all states that transition into this mask have already been processed
    # from smaller masks.
    for mask in range(1 << k):
        # Iterate through current key locations (0 to N-1)
        for u in range(N):
            if dp[mask][u] == float('inf'):
                continue # Cannot reach this state (mask, u)

            # From current location u, move to next key location v
            for v in range(N):
                # if u == v: continue # Moving from a key location to itself doesn't improve mask or distance

                move_dist = dists[u][v]
                if move_dist == float('inf'):
                    continue # Cannot move from u to v

                new_dist = dp[mask][u] + move_dist

                # Pruning: if new distance exceeds time limit T, this path segment makes the total path too long
                if new_dist > T:
                    continue

                # Calculate the new mask of visited candies.
                # This new mask is the union of:
                # 1. Candies visited up to the current state (mask)
                # 2. Candies visited on the shortest path segment from u to v (segment_candies_mask[u][v])
                # 3. Candy at destination v itself, if v is a candy location
                new_mask = mask
                new_mask |= segment_candies_mask[u][v]

                # If the destination v is a candy itself, it is now visited.
                # Candies are located at key_locations[1] through key_locations[k].
                # These correspond to original candy indices 0 through k-1.
                if 1 <= v <= k: # v is an index corresponding to a candy C_{v-1}
                     new_mask |= (1 << (v - 1)) # Add bit for candy C_{v-1}

                # Update DP table: minimum distance to reach state (new_mask, v)
                dp[new_mask][v] = min(dp[new_mask][v], new_dist)

    # Find the maximum number of visited candies while finishing at the goal square (key_idx N-1) within time T
    max_candies = -1
    goal_key_idx = N - 1 # G is the last element in key_locations

    # Check all possible masks
    for mask in range(1 << k):
        # If it's possible to reach the goal with this mask within time T
        if dp[mask][goal_key_idx] <= T:
            # Count the number of set bits in the mask
            num_candies_visited = bin(mask).count('1')
            # Update the maximum number of candies found
            max_candies = max(max_candies, num_candies_visited)

    print(max_candies)

# Execute the solve function
solve()