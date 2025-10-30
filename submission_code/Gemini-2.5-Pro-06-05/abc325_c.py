def solve():
    """
    Reads a grid of sensors and counts the number of connected groups.
    A group is defined by sensors that are connected horizontally, vertically,
    or diagonally, directly or indirectly. This is equivalent to finding the
    number of connected components in a graph.
    """
    try:
        # Read grid dimensions. Handles potential empty input lines for robustness.
        line = input()
        if not line:
            print(0)
            return
        H, W = map(int, line.split())
    except (ValueError, EOFError):
        # Catches malformed lines or end of file.
        print(0)
        return

    # Read the H lines of the grid.
    grid = [input() for _ in range(H)]

    # A 2D list to keep track of visited cells.
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    # Initialize the counter for sensor groups (connected components).
    count = 0

    # Iterate through each cell in the grid.
    for r in range(H):
        for c in range(W):
            # If we find a sensor ('#') that has not been part of a previously found group:
            if grid[r][c] == '#' and not visited[r][c]:
                # We've found a new group. Increment the counter.
                count += 1
                
                # Use an iterative Depth-First Search (DFS) to find all sensors
                # in the current group and mark them as visited.
                stack = [(r, c)]
                visited[r][c] = True
                
                while stack:
                    # Get the next cell to explore from the stack.
                    curr_r, curr_c = stack.pop()
                    
                    # Check all 8 neighbors (including diagonals).
                    for dr in range(-1, 2):  # Delta row: -1, 0, 1
                        for dc in range(-1, 2):  # Delta column: -1, 0, 1
                            # Skip the cell itself.
                            if dr == 0 and dc == 0:
                                continue
                            
                            next_r, next_c = curr_r + dr, curr_c + dc
                            
                            # Check if the neighbor is a valid, unvisited sensor.
                            if 0 <= next_r < H and 0 <= next_c < W and \
                               grid[next_r][next_c] == '#' and not visited[next_r][next_c]:
                                
                                # Mark as visited and add to the stack to explore its neighbors later.
                                visited[next_r][next_c] = True
                                stack.append((next_r, next_c))
                                
    # Print the total number of sensor groups found.
    print(count)

# Run the solution.
solve()