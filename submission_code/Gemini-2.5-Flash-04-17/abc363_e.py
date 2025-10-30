import heapq
import sys

def solve():
    # Read input
    H, W, Y = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, sys.stdin.readline().split())))

    # visited[r][c] will be True if the minimum sink level for cell (r, c) has been determined.
    visited = [[False] * W for _ in range(H)]
    
    # Priority queue stores tuples (sink_level, row, col)
    # It processes cells with lower sink levels first.
    pq = []

    # Add border cells to the priority queue.
    # The initial minimum sink level for a border cell is its own elevation,
    # as it is directly adjacent to the sea (at relative level 0).
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                heapq.heappush(pq, (grid[r][c], r, c))
                # We do NOT mark visited here. A cell is marked visited only when
                # it is extracted from the PQ, ensuring we process it with the
                # minimum possible sink level found so far.

    # We need to count how many cells sink at each specific level.
    # The maximum possible elevation is 10^5. The maximum possible minimum-maximum
    # path elevation (which is the sink level) is also 10^5.
    # We need counts for levels 1 up to 10^5. Array size 100001 covers indices 0..100000.
    # Levels are 1-based. sink_counts[k] will store count for level k.
    max_possible_sl = 100000 # Max A_ij is 10^5, max SL is 10^5.
    sink_counts = [0] * (max_possible_sl + 1) # Array size 100001 for levels 0..100000

    dr = [-1, 1, 0, 0] # Directions for neighbors: up, down, left, right
    dc = [0, 0, -1, 1]

    cells_processed = 0
    total_cells = H * W

    # Dijkstra-like process
    while pq:
        # Get the cell with the minimum sink level found so far
        level, r, c = heapq.heappop(pq)

        # If this cell has already been visited (meaning its minimum sink level
        # was already determined and processed), skip it.
        if visited[r][c]:
            continue

        # Mark the cell as visited. We have found its minimum sink level.
        visited[r][c] = True
        cells_processed += 1

        # Record that this cell sinks at this determined level.
        # Levels extracted from PQ are >= 1 (since A_ij >= 1).
        # The maximum possible sink level is bounded by max(A_ij) = 10^5.
        # So extracted 'level' will be between 1 and 100000.
        # We increment the count for this level.
        sink_counts[level] += 1 # level is guaranteed 1 <= level <= 100000

        # Optimization: If all cells are processed, we have found the sink level for every cell.
        # Any remaining items in the PQ are for cells already processed. We can stop.
        if cells_processed == total_cells:
             break

        # Explore the neighbors of the current cell
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check if the neighbor is within grid bounds and has not been visited yet
            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                 # The potential sink level for the neighbor (nr, nc) through (r, c)
                 # is the maximum of the level at which (r, c) sinks and the neighbor's elevation.
                 # This represents the bottleneck elevation on the path to the sea through (r, c).
                 potential_level = max(level, grid[nr][nc])
                 
                 # Push the neighbor into the priority queue with its potential minimum sink level
                 # The priority queue ensures we explore cells in increasing order of this potential level.
                 heapq.heappush(pq, (potential_level, nr, nc))

    # Compute the cumulative number of sunk cells up to each level.
    # cumulative_sunk[k] = total number of cells that sink at level <= k.
    # We need cumulative counts up to Y. Since Y <= 10^5, levels up to 10^5 are relevant.
    cumulative_sunk = [0] * (max_possible_sl + 1) # Array size 100001
    
    # cumulative_sunk[0] is 0.
    # cumulative_sunk[k] = cumulative_sunk[k-1] + sink_counts[k] for k >= 1.
    for level in range(1, max_possible_sl + 1):
        cumulative_sunk[level] = cumulative_sunk[level - 1] + sink_counts[level]

    # Calculate and print the remaining area for each year from 1 to Y.
    total_area = H * W

    for year in range(1, Y + 1):
        # After 'year' years, the sea level is 'year'.
        # The cells that have sunk are all cells with a sink level <= 'year'.
        # The number of sunk cells is cumulative_sunk[year].
        
        # The constraint Y <= 10^5 means year <= 100000.
        # So cumulative_sunk[year] is always a valid index (1 <= year <= 100000).
        sunk_by_year = cumulative_sunk[year]
        
        remaining_area = total_area - sunk_by_year
        print(remaining_area)

solve()