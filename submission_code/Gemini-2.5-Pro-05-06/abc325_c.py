import collections
import sys

def main():
    # Read H, W from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid S
    # S will be a list of strings. S[i] is the i-th row. S[i][j] is the character at (i,j).
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # visited[r][c] will be True if the sensor at (r,c) has been visited as part of a component,
    # False otherwise. Initialize all to False.
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    component_count = 0

    # Define the changes in row and column for the 8 possible directions
    # (dr, dc) pairs:
    # (-1, -1) (up-left), (-1, 0) (up), (-1, 1) (up-right)
    # ( 0, -1) (left),                     ( 0, 1) (right)
    # ( 1, -1) (down-left), ( 1, 0) (down), ( 1, 1) (down-right)
    dr_offsets = [-1, -1, -1, 0, 0, 1, 1, 1]
    dc_offsets = [-1, 0, 1, -1, 1, -1, 0, 1]

    # Iterate through each cell of the grid
    for r_start in range(H):
        for c_start in range(W):
            # If we find a sensor ('#') that hasn't been visited yet,
            # it signifies the start of a new connected component.
            if S[r_start][c_start] == '#' and not visited[r_start][c_start]:
                component_count += 1
                
                # Start a Breadth-First Search (BFS) to find all sensors in this component.
                # Initialize a queue for BFS and add the starting sensor.
                q = collections.deque()
                q.append((r_start, c_start))
                visited[r_start][c_start] = True # Mark the starting sensor as visited.

                while q:
                    curr_r, curr_c = q.popleft() # Get the current sensor to explore from.

                    # Explore all 8 neighbors of the current sensor.
                    for i in range(8):
                        nr, nc = curr_r + dr_offsets[i], curr_c + dc_offsets[i]

                        # Check if the neighbor's coordinates are within the grid boundaries.
                        if 0 <= nr < H and 0 <= nc < W:
                            # If the neighbor is a sensor and has not been visited yet:
                            if S[nr][nc] == '#' and not visited[nr][nc]:
                                visited[nr][nc] = True # Mark it as visited.
                                q.append((nr, nc)) # Add it to the queue to explore its neighbors later.
    
    # Print the total number of connected components found.
    sys.stdout.write(str(component_count) + "
")

if __name__ == '__main__':
    main()