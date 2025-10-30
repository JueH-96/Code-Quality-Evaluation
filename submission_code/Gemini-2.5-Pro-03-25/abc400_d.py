# YOUR CODE HERE
import collections
import sys

# Function to solve the problem
def solve():
    # Read grid dimensions H (height/rows) and W (width/columns)
    H, W = map(int, sys.stdin.readline().split())
    
    # Read start coordinates (A, B) and target coordinates (C, D) from the same line
    # The input coordinates are 1-based.
    line_coords = list(map(int, sys.stdin.readline().split()))
    A, B = line_coords[0], line_coords[1]
    C, D = line_coords[2], line_coords[3]

    # Convert coordinates to 0-based indexing for array access
    start_r, start_c = A - 1, B - 1
    target_r, target_c = C - 1, D - 1
    
    # Read the grid layout line by line
    # S is a list of strings, where S[i] is the i-th row of the grid.
    # '.' represents a road, '#' represents a wall.
    S = [sys.stdin.readline().strip() for _ in range(H)] 

    # Initialize a 2D array `dist` to store the minimum number of kicks required to reach each cell.
    # Initialize all distances to infinity, representing unreachability initially.
    dist = [[float('inf')] * W for _ in range(H)]
    
    # The number of kicks required to reach the starting cell is 0.
    dist[start_r][start_c] = 0

    # Initialize a double-ended queue (deque) for the 0-1 Breadth-First Search (BFS).
    # The deque will store tuples of (row, column) coordinates of cells to visit.
    dq = collections.deque([(start_r, start_c)])

    # Perform the 0-1 BFS to find the shortest path in terms of number of kicks
    while dq:
        # Retrieve the cell from the front of the deque.
        # In 0-1 BFS, cells reachable with fewer cost=1 edges (kicks) are processed earlier.
        r, c = dq.popleft()
        # Get the minimum number of kicks found so far to reach this cell (r, c).
        k = dist[r][c] 

        # Check if the cell retrieved is the target cell.
        if r == target_r and c == target_c:
             # If the target cell is popped, its distance `k` is the minimum required kicks.
             print(k) # Print the result
             return   # Exit the function as the solution is found

        # Explore neighbors reachable by standard movement (cost = 0 kicks).
        # This involves moving to adjacent (up, down, left, right) road cells.
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # Relative coordinates for neighbors
            nr, nc = r + dr, c + dc # Calculate absolute neighbor coordinates
            
            # Check if the neighbor coordinates are within the grid boundaries (0 <= row < H, 0 <= col < W).
            if 0 <= nr < H and 0 <= nc < W:
                 # Check if the neighbor cell is a road (represented by '.').
                 if S[nr][nc] == '.':
                    # If visiting this neighbor through cell (r, c) requires `k` kicks,
                    # check if this is better (fewer kicks) than the current known minimum kicks to reach (nr, nc).
                    # Since movement costs 0 kicks, the cost remains `k`.
                    if dist[nr][nc] > k: 
                        # Update the minimum kicks for the neighbor cell.
                        dist[nr][nc] = k
                        # Add the neighbor cell to the front of the deque. This is characteristic of 0-1 BFS for 0-cost edges.
                        dq.appendleft((nr, nc)) 

        # Explore cells potentially reachable after using a front kick (cost = 1 kick).
        # A front kick action allows potentially breaking walls within 2 steps in a chosen direction.
        # This fundamentally changes reachability. We model this by considering that after a kick action
        # potentially performed near cell (r, c), any cell within a 5x5 square centered at (r, c) might become reachable.
        # This transition effectively costs 1 kick.
        # Check all cells (tr, tc) where the row `tr` is in the range [r-2, r+2] and column `tc` is in the range [c-2, c+2].
        for dr_kick in range(-2, 3): # Iterate through row offsets from -2 to +2
            for dc_kick in range(-2, 3): # Iterate through column offsets from -2 to +2
                # Calculate coordinates of the potential cell reachable after a kick possibly followed by moves.
                tr, tc = r + dr_kick, c + dc_kick
                
                # Check if the potential cell coordinates are within the grid boundaries.
                if 0 <= tr < H and 0 <= tc < W:
                    # Reaching cell (tr, tc) via a path involving one more kick (total k+1 kicks)
                    # might be possible. Check if this path offers fewer kicks than the current known minimum for (tr, tc).
                    if dist[tr][tc] > k + 1:
                        # Update the minimum kicks required to reach cell (tr, tc).
                        dist[tr][tc] = k + 1
                        # Add the cell (tr, tc) to the back of the deque. This is standard for 1-cost edges in 0-1 BFS.
                        dq.append((tr, tc)) 

    # If the while loop completes, it means the deque became empty, 
    # but the target cell was never popped from the deque. This implies the target is unreachable from the start.
    print("-1")

# Call the solve function to execute the program logic
solve()