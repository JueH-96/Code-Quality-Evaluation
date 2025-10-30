# YOUR CODE HERE
import sys
import heapq

# Read input using fast I/O
readline = sys.stdin.readline

def solve():
    """
    Solves the island sinking problem. Reads grid dimensions, number of years, and elevation map.
    Calculates the remaining island area for each year from 1 to Y.
    """
    H, W, Y = map(int, readline().split())
    
    A = []
    for _ in range(H):
        A.append(list(map(int, readline().split())))

    # Initialize data structures for Dijkstra-like algorithm.
    # min_max_elev[r][c] will store the minimum possible maximum elevation M(r,c) 
    # on any path from a boundary cell to cell (r,c).
    # Initialize all path costs to infinity.
    min_max_elev = [[float('inf')] * W for _ in range(H)]
    
    # Priority queue stores tuples: (current_path_max_elevation, row, col).
    # It's a min-heap ordered by the path maximum elevation (cost).
    pq = [] 

    max_A_val = 0 # Track the maximum elevation on the island to determine array sizes later.

    # Initialize the algorithm starting from all boundary cells.
    for r in range(H):
        for c in range(W):
            # Keep track of the overall maximum elevation value found in the grid.
            if A[r][c] > max_A_val:
                 max_A_val = A[r][c]

            # Check if the cell is on the boundary of the grid (first/last row or first/last column).
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                # For boundary cells, the path to the boundary is trivial: just the cell itself.
                # The maximum elevation on this path is its own elevation.
                # This is the initial cost/distance for boundary cells.
                min_max_elev[r][c] = A[r][c]
                # Add the boundary cell to the priority queue to start the exploration process.
                heapq.heappush(pq, (A[r][c], r, c))

    # Run the Dijkstra-like algorithm. This computes the minimum bottleneck capacity path cost (M(r,c))
    # from the boundary to every cell (r,c).
    while pq:
        # Extract the cell with the minimum path maximum elevation found so far.
        cost, r, c = heapq.heappop(pq)

        # Optimization: If we've found a path with a smaller maximum elevation to this cell already,
        # then this entry in the priority queue is outdated. Skip processing it.
        if cost > min_max_elev[r][c]:
            continue
        
        # Explore the neighbors of the current cell (up, down, left, right).
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check if the neighbor coordinates are within the valid grid boundaries.
            if 0 <= nr < H and 0 <= nc < W:
                # Calculate the maximum elevation encountered on the path extending to the neighbor (nr, nc)
                # through the current cell (r, c). This path's bottleneck capacity is determined by the maximum of:
                # 1. The bottleneck capacity of the path to the current cell `cost`.
                # 2. The elevation of the neighbor cell itself `A[nr][nc]`.
                new_cost = max(cost, A[nr][nc])
                
                # Relaxation step: If this new path to (nr, nc) has a smaller maximum elevation 
                # than any path found so far to (nr, nc), update the minimum max elevation for (nr, nc)
                # and add/update its entry in the priority queue.
                if new_cost < min_max_elev[nr][nc]:
                    min_max_elev[nr][nc] = new_cost
                    heapq.heappush(pq, (new_cost, nr, nc))

    
    # Post-processing phase: Calculate the count of cells sinking at each specific year.
    # The maximum possible sinking time for any cell is bounded by max_A_val.
    # We need an array to store counts for years up to max_A_val.
    count_size = max_A_val + 1
    # sunk_count_at_year[k] will store the number of cells that sink exactly at year k.
    sunk_count_at_year = [0] * count_size

    # Iterate through all cells in the grid to determine their individual sinking times.
    for r in range(H):
        for c in range(W):
             # A cell (r,c) sinks at the minimum year L such that both conditions are met:
             # 1. The cell's elevation is not greater than the sea level: A[r][c] <= L.
             # 2. The cell is connected to the sea (boundary) via a path where all cells 
             #    have elevation not greater than L. This requires L >= M(r,c).
             # Combining these, the minimum L is max(A[r][c], M(r,c)). M(r,c) is stored in min_max_elev[r][c].
             # If min_max_elev[r][c] remained infinity, it implies the cell is unreachable from the boundary,
             # thus it never sinks. The sink time would be effectively infinite.
             sink_time = max(A[r][c], min_max_elev[r][c])
             
             # We only care about finite sinking times within the range [1, max_A_val].
             # Check if sink_time is finite and within the bounds of our count array.
             # `sink_time < count_size` handles both finite check and boundary check.
             if sink_time < count_size : 
                # Sink time must be an integer, elevations are integers. Cast for safety.
                sunk_count_at_year[int(sink_time)] += 1 

    # Calculate and collect the remaining island area for each year from 1 to Y.
    current_total_sunk = 0 # This will keep track of the cumulative number of cells sunk up to the current year.
    initial_area = H * W # The total number of land cells at the start.
    
    results = [] # List to store the results (remaining area for each year) efficiently as strings.
    
    # Iterate through each year from 1 to Y.
    for k in range(1, Y + 1):
        # Add the count of cells that sink exactly at year k to the running total count of sunk cells.
        # We need to check if k is a valid index in our sunk_count_at_year array (i.e., k <= max_A_val).
        if k < count_size: 
            current_total_sunk += sunk_count_at_year[k]
        
        # The remaining area at year k is the initial total area minus the total number of cells sunk by year k.
        area_k = initial_area - current_total_sunk
        # Append the result for year k to our results list. Convert to string for efficient joining later.
        results.append(str(area_k)) 

    # Print all collected results, each on a new line. Using "
".join is generally faster than multiple print calls.
    print("
".join(results))

# Execute the main solution function when the script is run.
solve()

# END OF YOUR CODE HERE