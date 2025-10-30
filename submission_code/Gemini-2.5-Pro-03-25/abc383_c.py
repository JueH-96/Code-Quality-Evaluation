# YOUR CODE HERE
import sys
import collections

# Function to potentially speed up input reading using sys.stdin.readline
# This can be helpful for large inputs, although standard input() might be sufficient for these constraints.
def get_ints():
    """Reads a line of space-separated integers from stdin."""
    return map(int, sys.stdin.readline().strip().split())

def get_str():
    """Reads a line as a string from stdin."""
    return sys.stdin.readline().strip()

def solve():
    """Solves the AtCoder humidified cells problem."""
    
    # Read grid dimensions H (height/rows), W (width/columns) and maximum distance D
    H, W, D = get_ints()
    
    # Read the grid state as a list of strings
    S = [get_str() for _ in range(H)]

    # Initialize distance matrix `dist` with -1. 
    # -1 indicates that the cell is unvisited or unreachable.
    dist = [[-1] * W for _ in range(H)]
    
    # Initialize a double-ended queue `q` for BFS.
    # Using collections.deque is efficient for queue operations (append and popleft).
    q = collections.deque()

    # Find all initial humidifier locations ('H').
    # Set their distance to 0 and add them to the queue as starting points for the BFS.
    # We iterate through the grid to find all 'H' cells.
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'H':
                # Check if this cell has already been processed (e.g., potentially if input format allowed duplicates, though unlikely).
                # This ensures each source starts with distance 0 correctly.
                if dist[r][c] == -1: 
                   dist[r][c] = 0
                   q.append((r, c)) # Add (row, col) tuple to the queue

    # Define the relative coordinates for moving to adjacent cells: Up, Down, Left, Right.
    dr = [-1, 1, 0, 0] # Change in row for Up, Down
    dc = [0, 0, -1, 1] # Change in column for Left, Right

    # Perform Breadth-First Search (BFS) starting from all humidifier cells simultaneously.
    while q:
        # Get the current cell coordinates (r, c) from the front of the queue.
        r, c = q.popleft()
        
        # Get the distance of the current cell from the nearest humidifier.
        current_dist = dist[r][c]

        # Optimization: If the current cell's distance is already D, we don't need to explore its neighbors.
        # Any neighbor reached from this cell would have a distance of D+1, which is greater than the maximum allowed distance D.
        # This optimization can prune the search space significantly if D is small compared to the grid size.
        if current_dist == D:
            continue 

        # Explore the 4 adjacent neighbors (Up, Down, Left, Right).
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i] # Calculate neighbor coordinates
            
            # Check if the neighbor's coordinates (nr, nc) are within the grid boundaries.
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the neighbor cell is not a wall ('#'). Walls block movement.
                if S[nr][nc] != '#':
                    # Check if the neighbor cell has not been visited yet (its distance is still -1).
                    # BFS guarantees that the first time we reach a cell, it's via a shortest path from the sources.
                    if dist[nr][nc] == -1:
                        # Calculate the distance to the neighbor: current cell's distance + 1.
                        new_dist = current_dist + 1
                        # Update the distance for the neighbor cell.
                        dist[nr][nc] = new_dist
                        # Add the neighbor cell to the queue for further exploration.
                        q.append((nr, nc))
                        
    # After the BFS is complete, count the number of humidified floor cells.
    humidified_count = 0
    for r in range(H):
        for c in range(W):
            # A cell is considered humidified if it meets two conditions:
            # 1. It is not a wall ('#'). This includes both empty floor cells ('.') and cells with humidifiers ('H').
            #    The problem states 'H' cells are humidifiers placed on floor cells, implying they are floor cells too.
            if S[r][c] != '#':
                # 2. It is reachable from at least one humidifier within D steps.
                #    This means its distance `dist[r][c]` is not -1 (i.e., it was visited).
                #    Due to the BFS optimization (`if current_dist == D: continue`), any cell that has been visited (`dist[r][c] != -1`)
                #    is guaranteed to have a distance `dist[r][c] <= D`.
                if dist[r][c] != -1:
                    humidified_count += 1

    # Print the final count of humidified floor cells to standard output.
    print(humidified_count)

# Call the solve function to execute the logic for the problem.
solve()