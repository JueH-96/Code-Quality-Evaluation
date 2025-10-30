import collections
import sys

def solve():
    # Read H and W from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid rows into a list of strings
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip()) # .strip() removes the newline character

    # The target character sequence we need to follow
    target_chars = "snuke"
    
    # Initial check: The character at the starting cell (0,0) must be 's'.
    # If not, no valid path can start, so we print "No" and exit.
    if grid[0][0] != 's':
        print("No")
        return

    # Initialize a deque for Breadth-First Search (BFS).
    # Each element in the queue will be a tuple: (row, col, char_idx).
    # 'row' and 'col' are the 0-indexed coordinates of the current cell.
    # 'char_idx' is the 0-indexed position in 'target_chars' that the current cell's character
    # should match. For the starting cell (0,0), its character 's' is at index 0 of "snuke".
    q = collections.deque()
    q.append((0, 0, 0)) 

    # Initialize a set to keep track of visited states.
    # A state is defined by (row, col, char_idx). We can visit the same (row, col) cell
    # multiple times, but only if the required 'char_idx' for that visit is different.
    visited = set()
    visited.add((0, 0, 0))

    # Define possible movements: up, down, left, right
    dr = [-1, 1, 0, 0] # changes in row
    dc = [0, 0, -1, 1] # changes in column

    # Start the BFS loop
    while q:
        r, c, char_idx = q.popleft()

        # If the current cell is the destination cell (H-1, W-1),
        # and we reached it with a character matching the sequence,
        # then a valid path exists. Print "Yes" and terminate.
        if r == H - 1 and c == W - 1:
            print("Yes")
            return

        # Calculate the index of the character that the *next* cell in the path must match.
        # This wraps around using modulo 5 (since "snuke" has 5 characters).
        next_char_idx = (char_idx + 1) % 5

        # Explore all four adjacent neighbors
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i] # New row and column coordinates for the neighbor

            # Check if the neighbor is within the grid boundaries
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the character in the neighbor cell matches the required character
                if grid[nr][nc] == target_chars[next_char_idx]:
                    # Check if this specific state (neighbor cell and its required char_idx)
                    # has been visited before. If not, add it to visited and enqueue it.
                    if (nr, nc, next_char_idx) not in visited:
                        visited.add((nr, nc, next_char_idx))
                        q.append((nr, nc, next_char_idx))
    
    # If the queue becomes empty and we have not found a path to the destination cell,
    # it means no such path exists.
    print("No")

# Call the solve function to execute the program
solve()