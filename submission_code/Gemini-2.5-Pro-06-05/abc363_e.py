import sys
import heapq

def solve():
    """
    Reads island dimensions, elevations, and years, then calculates and prints
    the remaining island area for each year.
    """
    # Use fast I/O for performance with large inputs.
    try:
        line = sys.stdin.readline()
        if not line:
            return  # Handle empty input
        H, W, Y = map(int, line.split())
        
        A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    except (IOError, ValueError):
        # Gracefully exit on malformed or empty input.
        return

    # Constants based on problem constraints.
    # Max elevation/year can be 100,000.
    MAX_VAL = 100000
    # A value representing infinity, larger than any possible sink year.
    inf = MAX_VAL + 2
    
    # Priority queue for Dijkstra's algorithm: stores (cost, r, c).
    pq = []
    
    # sink_year[r][c] will store the minimum year for cell (r, c) to sink.
    sink_year = [[inf] * W for _ in range(H)]

    # Initialize Dijkstra's from all border cells, as they are adjacent to the sea.
    # The cost to flood a border cell is its own elevation.
    for r in range(H):
        for c in range(W):
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                sink_year[r][c] = A[r][c]
                heapq.heappush(pq, (A[r][c], r, c))

    # Directions for moving to adjacent cells (up, down, left, right).
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # Run Dijkstra's algorithm to find the sink year for every cell.
    while pq:
        year, r, c = heapq.heappop(pq)

        # If we have found a better (earlier) path to this cell, skip.
        if year > sink_year[r][c]:
            continue

        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < H and 0 <= nc < W:
                # The sea must overcome the highest elevation on a path to a cell.
                # This is the "bottleneck" of the path.
                new_year = max(year, A[nr][nc])
                
                # If we found a path that floods the cell earlier, update it.
                if new_year < sink_year[nr][nc]:
                    sink_year[nr][nc] = new_year
                    heapq.heappush(pq, (new_year, nr, nc))

    # --- Part 2: Answering Queries ---
    
    # Count how many cells sink in each specific year.
    # The array needs to be large enough for both the max elevation and max query year.
    max_year_to_consider = max(Y, MAX_VAL)
    sunk_per_year = [0] * (max_year_to_consider + 1)
    
    for r in range(H):
        for c in range(W):
            # All cells are part of one island, so they will eventually sink.
            # The sink year will be at most MAX_VAL.
            s_y = sink_year[r][c]
            sunk_per_year[s_y] += 1
            
    # Calculate and print the remaining area for each year from 1 to Y.
    remaining_area = H * W
    for y in range(1, Y + 1):
        # In year y, the cells with sink_year==y get submerged.
        remaining_area -= sunk_per_year[y]
        print(remaining_area)

if __name__ == "__main__":
    solve()