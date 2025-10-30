import collections
import sys

def solve():
    # Read H, W, D from the first line
    H, W, D = map(int, sys.stdin.readline().split())
    
    # Read the grid S
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # dist[r][c] will store the minimum distance from a humidifier to (r,c).
    # Initialize with -1, indicating that the cell has not been visited yet.
    dist = [[-1] * W for _ in range(H)]

    # Queue for BFS, storing (row, col) coordinates of cells to visit.
    q = collections.deque()

    # Set to store all unique humidified floor cells.
    # A cell is a floor cell if its original character was '.' or 'H'.
    # This set will correctly accumulate all such cells reachable within distance D.
    humidified_cells = set()

    # Define directions for moving (up, down, left, right)
    dr = [-1, 1, 0, 0]  # Row changes
    dc = [0, 0, -1, 1]  # Column changes

    # Step 1: Initialize BFS with all humidifier cells ('H')
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'H':
                q.append((r, c))      # Add to queue
                dist[r][c] = 0        # Distance from itself is 0
                humidified_cells.add((r, c)) # 'H' cells are always humidified floor cells

    # Step 2: Perform BFS
    while q:
        r, c = q.popleft() # Get the current cell

        # If the current cell is already at the maximum allowed distance D,
        # it cannot humidify any new cells because any further move would exceed D.
        if dist[r][c] == D:
            continue

        # Explore all four neighbors
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i] # Calculate neighbor coordinates

            # Check if the neighbor is within grid boundaries
            if not (0 <= nr < H and 0 <= nc < W):
                continue

            # Check if the neighbor is a wall ('#'). Walls cannot be passed through.
            if S[nr][nc] == '#':
                continue

            # If the neighbor has not been visited yet (dist[nr][nc] == -1)
            # This ensures we find the shortest path in an unweighted graph.
            if dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1 # Update distance
                q.append((nr, nc))           # Add to queue for further exploration
                
                # Add this newly humidified floor cell to the set.
                # Since we already checked that S[nr][nc] is not '#',
                # it must be either '.' or 'H', hence a floor cell.
                humidified_cells.add((nr, nc))

    # Step 3: Print the total count of unique humidified floor cells
    sys.stdout.write(str(len(humidified_cells)) + '
')

# Call the solve function to execute the program
solve()