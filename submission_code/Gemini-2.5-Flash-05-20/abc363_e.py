import sys
import heapq

def solve():
    H, W, Y = map(int, sys.stdin.readline().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, sys.stdin.readline().split())))

    # min_sink_level[r][c] stores the minimum sea level at which cell (r,c) sinks.
    # Initialize with infinity for all cells.
    min_sink_level = [[float('inf')] * W for _ in range(H)]

    # Priority queue stores tuples of (level, r, c)
    # The 'level' indicates the sea level at which (r,c) would sink.
    pq = []

    # Add all border cells to the priority queue
    # These cells are adjacent to the "initial sea" and will sink when sea level
    # reaches their elevation.
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                min_sink_level[r][c] = A[r][c]
                heapq.heappush(pq, (A[r][c], r, c))

    # max_possible_elevation is 10^5 (from constraints)
    max_elevation = 10**5 
    # yearly_sunk_count[k] stores how many new cells sink *at* sea level k
    yearly_sunk_count = [0] * (max_elevation + 1)

    # Directions for neighbors (up, down, left, right)
    dr = [-1, 1, 0, 0] 
    dc = [0, 0, -1, 1] 

    # Dijkstra's algorithm
    # This finds the minimum sea level for each cell to sink.
    while pq:
        level, r, c = heapq.heappop(pq)

        # If we have already found a way to sink this cell at an earlier level, skip.
        # This handles cases where a cell might be pushed multiple times with different levels.
        if level > min_sink_level[r][c]:
            continue

        # This cell (r,c) sinks when the sea level reaches 'level'.
        # Increment the count for cells sinking at this specific level.
        # Ensure level is within valid bounds for yearly_sunk_count array.
        if level <= max_elevation:
            yearly_sunk_count[level] += 1
        # Else, if level > max_elevation, it means this cell sinks at a level beyond the max_elevation
        # given in constraints, which might happen if A_ij could be higher.
        # For this problem, A_ij <= 10^5, so level will not exceed max_elevation by much,
        # but max(level, A[nr][nc]) could potentially make it go to 10^5 + epsilon for a very specific setup.
        # However, for this problem, levels will stay within the 1 to 10^5 range.

        # Explore neighbors
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]

            # Check if neighbor is within grid boundaries
            if 0 <= nr < H and 0 <= nc < W:
                # Calculate the sea level required for the neighbor (nr, nc) to sink via (r,c).
                # It's the maximum of the current 'level' (water level that reached r,c)
                # and the neighbor's own elevation A[nr][nc].
                new_level = max(level, A[nr][nc])

                # If this new_level is lower than the previously recorded min_sink_level for (nr,nc),
                # it means we found a new, earlier way for (nr,nc) to sink.
                if new_level < min_sink_level[nr][nc]:
                    min_sink_level[nr][nc] = new_level
                    heapq.heappush(pq, (new_level, nr, nc))

    # Calculate remaining land area for each year
    current_remaining_area = H * W
    results = []

    # Iterate from year 1 up to Y
    for y in range(1, Y + 1):
        # Subtract cells that sink at sea level 'y' (i.e., at year 'y')
        # Check if 'y' is within the pre-calculated range of sinking levels.
        # For y > max_elevation, yearly_sunk_count[y] would be 0, correctly contributing no more sinking.
        if y <= max_elevation:
            current_remaining_area -= yearly_sunk_count[y]
        
        results.append(current_remaining_area)
    
    # Print results
    for res in results:
        sys.stdout.write(str(res) + '
')

solve()