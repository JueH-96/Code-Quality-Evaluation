import sys
import collections

def solve():
    H, W, Y = map(int, sys.stdin.readline().split())
    
    A = []
    max_A_val = 0
    for _ in range(H):
        row = list(map(int, sys.stdin.readline().split()))
        A.append(row)
        for elevation_val in row:
            if elevation_val > max_A_val:
                max_A_val = elevation_val
    
    # S_sink[r][c] will store the minimum sea level at which cell (r,c) sinks.
    # Initialize with a value larger than any possible S_sink value.
    # Max possible S_sink is max_A_val. So max_A_val + 1 can represent infinity.
    infinity = max_A_val + 1 
    S_sink = [[infinity] * W for _ in range(H)]

    # Bucket-based Dijkstra
    # buckets[k] stores a deque of cells (r,c) whose current tentative S_sink value is k.
    # Buckets are indexed from 0 to max_A_val.
    buckets = [collections.deque() for _ in range(max_A_val + 1)]

    # Initialize S_sink values for boundary cells and add them to buckets.
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                elevation = A[r][c]
                S_sink[r][c] = elevation
                # elevation is A[r][c], so it's <= max_A_val and a valid index for buckets.
                buckets[elevation].append((r, c))
    
    # Dijkstra main loop using buckets
    # Iterate through buckets for increasing costs (sea levels).
    for cost_s in range(max_A_val + 1): # cost_s is the current sea level being processed.
                                        # Min elevation is 1, so buckets[0] will remain empty.
        while buckets[cost_s]:
            r, c = buckets[cost_s].popleft()

            # If cost_s > S_sink[r][c], this means (r,c) was already processed via a path
            # that sinks it at an even lower sea level (S_sink[r][c]). So, skip this stale entry.
            if cost_s > S_sink[r][c]:
                continue
            
            # Explore neighbors (up, down, left, right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                if 0 <= nr < H and 0 <= nc < W: # Check if neighbor is within grid boundaries
                    # The sea level required to sink (nr,nc) via path through (r,c)
                    # is the maximum of current path's requirement (cost_s) 
                    # and neighbor's own elevation (A[nr][nc]).
                    new_cost_for_neighbor = max(cost_s, A[nr][nc])
                    
                    # If this path offers a lower sinking sea level for (nr,nc)
                    if new_cost_for_neighbor < S_sink[nr][nc]:
                        S_sink[nr][nc] = new_cost_for_neighbor
                        # Add (nr,nc) to the bucket corresponding to its new S_sink value.
                        # new_cost_for_neighbor must be <= max_A_val because both
                        # cost_s and A[nr][nc] are <= max_A_val.
                        buckets[new_cost_for_neighbor].append((nr, nc))

    # After Dijkstra, S_sink[r][c] holds the exact sea level at which cell (r,c) sinks.
    # All cells are connected, so S_sink[r][c] will be <= max_A_val for all (r,c).
    
    # delta_sunk_count[k] = number of cells that sink exactly when sea level becomes k.
    # Array size is max_A_val + 1 for indices 0...max_A_val.
    delta_sunk_count = [0] * (max_A_val + 1)
    for r in range(H):
        for c in range(W):
            delta_sunk_count[S_sink[r][c]] += 1
    
    # Calculate and print results for Y years
    results = []
    num_sunk_cells_total = 0
    initial_land_area = H * W

    for year_k in range(1, Y + 1):
        # If current year (sea level) is within the range of elevations,
        # add newly sunk cells for this sea level.
        if year_k <= max_A_val:
            num_sunk_cells_total += delta_sunk_count[year_k]
        
        # If year_k > max_A_val, all cells that can sink (up to max_A_val) have already sunk.
        # num_sunk_cells_total will not change further.
        
        results.append(str(initial_land_area - num_sunk_cells_total))

    sys.stdout.write("
".join(results) + "
")

solve()