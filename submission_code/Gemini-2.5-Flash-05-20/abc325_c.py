import collections
import sys

def solve():
    # Read H and W from the first line
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid row by row
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())

    # Initialize a 2D array to keep track of visited cells
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    # Initialize the counter for connected components
    num_components = 0

    # Define the 8 possible directions for neighbors (horizontal, vertical, and diagonal)
    # dr: change in row, dc: change in column
    dr = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc = [-1, 0, 1, -1, 1, -1, 0, 1]

    # Iterate over each cell in the grid
    for r in range(H):
        for c in range(W):
            # If the current cell contains a sensor ('#') and has not been visited yet
            if grid[r][c] == '#' and not visited[r][c]:
                # Increment the component count, as we found a new unvisited component
                num_components += 1
                
                # Start a Breadth-First Search (BFS) from this sensor
                q = collections.deque([(r, c)])
                visited[r][c] = True # Mark the starting cell as visited

                while q:
                    curr_r, curr_c = q.popleft() # Get the current cell from the queue

                    # Explore all 8 neighbors of the current cell
                    for i in range(8):
                        next_r, next_c = curr_r + dr[i], curr_c + dc[i]

                        # Check if the neighbor is within the grid boundaries
                        if 0 <= next_r < H and 0 <= next_c < W:
                            # Check if the neighbor is a sensor ('#') and has not been visited
                            if grid[next_r][next_c] == '#' and not visited[next_r][next_c]:
                                visited[next_r][next_c] = True # Mark the neighbor as visited
                                q.append((next_r, next_c)) # Add the neighbor to the queue for further exploration
                                
    # Print the total number of connected sensor components
    print(num_components)

# Ensure the solve function is called when the script is executed
if __name__ == '__main__':
    solve()