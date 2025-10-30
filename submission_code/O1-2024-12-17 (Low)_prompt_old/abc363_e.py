def solve():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    H, W, Y = map(int, input_data[:3])
    A_list = input_data[3:]
    
    # Read elevations into a 2D list A.
    # We'll use 0-based indexing for convenience.
    A = [list(map(int, A_list[r*W : (r+1)*W])) for r in range(H)]
    
    # We will assign each cell a "sink_year": the earliest year at which it becomes submerged.
    # This is determined by a multi-source priority queue (min-heap) search from the boundary.
    # boundary cells' initial sink_year is their own elevation, because:
    # they sink as soon as sea level >= elevation_of_boundary_cell.
    # Then we propagate inward, enforcing sink_year[nbr] = max(sink_year[cell], elevation[nbr]).
    
    sink_year = [[0]*W for _ in range(H)]
    visited = [[False]*W for _ in range(H)]
    pq = []  # (year_submerged, row, col)
    
    # Push boundary cells into the min-heap
    # Top row and bottom row
    for j in range(W):
        # top row: i=0
        visited[0][j] = True
        sink_year[0][j] = A[0][j]
        heapq.heappush(pq, (sink_year[0][j], 0, j))
        # bottom row: i=H-1 (if H>1, to avoid double-pushing single-row case)
        if H > 1:
            visited[H-1][j] = True
            sink_year[H-1][j] = A[H-1][j]
            heapq.heappush(pq, (sink_year[H-1][j], H-1, j))
    
    # Left column and right column (excluding corners already done above)
    for i in range(1, H-1):
        visited[i][0] = True
        sink_year[i][0] = A[i][0]
        heapq.heappush(pq, (sink_year[i][0], i, 0))
        if W > 1:
            visited[i][W-1] = True
            sink_year[i][W-1] = A[i][W-1]
            heapq.heappush(pq, (sink_year[i][W-1], i, W-1))
    
    # Directions for neighbors
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # D'Johnson / BFS-with-heap approach
    while pq:
        y_sub, r, c = heapq.heappop(pq)
        # If this is out-of-date info for (r,c), skip
        if y_sub > sink_year[r][c]:
            continue
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                visited[nr][nc] = True
                # This neighbor will sink at the max of "current sink_year" and its own elevation
                new_sink = max(y_sub, A[nr][nc])
                sink_year[nr][nc] = new_sink
                heapq.heappush(pq, (new_sink, nr, nc))
    
    # Now sink_year[i][j] tells us the earliest year that cell (i,j) goes under.
    # We count how many cells sink at each possible year (capped at 100000, since A[i][j] ≤ 100000).
    
    MAX_ELEV = 100000
    freq = [0] * (MAX_ELEV+1)  # freq[x] = number of cells that sink exactly at year x
    total_cells = H*W
    
    for i in range(H):
        for j in range(W):
            y = sink_year[i][j]
            if y > MAX_ELEV:
                y = MAX_ELEV
            freq[y] += 1
    
    # Build a prefix sum: prefix_sunk[x] = number of cells that sink on or before year x
    prefix_sunk = [0] * (MAX_ELEV+1)
    running = 0
    for x in range(MAX_ELEV+1):
        running += freq[x]
        prefix_sunk[x] = running
    
    # For each year 1..Y, number of sunken cells = prefix_sunk[min(year, MAX_ELEV)]
    # area = total_cells - that number
    import sys
    out = []
    for year in range(1, Y+1):
        idx = year if year <= MAX_ELEV else MAX_ELEV
        sunk_count = prefix_sunk[idx]
        area_remain = total_cells - sunk_count
        out.append(str(area_remain))
    
    print("
".join(out))

# For local testing (not part of submission), you can uncomment:
# if __name__ == "__main__":
#     solve()