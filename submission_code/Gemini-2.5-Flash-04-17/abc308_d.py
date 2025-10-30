# YOUR CODE HERE
import collections
import sys

# Read H and W
H, W = map(int, sys.stdin.readline().split())

# Read the grid
# Store the grid as a list of strings for efficient row access and char access by index
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Target sequence "snuke"
target = "snuke"

# Check the starting cell (0,0) - which is (1,1) in problem statement's 1-based indexing.
# The first cell must match the first character of "snuke", which is 's'.
if grid[0][0] != target[0]:
    print("No")
    sys.exit() # Exit immediately as a valid path cannot begin from (0,0)

# Initialize the visited array.
# visited[r][c][k] is a boolean indicating whether the state (r, c, k) has been visited.
# A state (r, c, k) means we have reached cell (r, c), and this cell (r, c)
# contains the character target[k], following the "snuke" pattern.
# The dimensions are H rows, W columns, and 5 possible character indices (0 to 4).
visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]

# Initialize the queue for Breadth-First Search (BFS).
# A deque is used for efficient appending and popping from both ends.
q = collections.deque()

# Define the starting state.
# The path starts at (0,0), and this cell corresponds to the 0-th character ('s')
# in the "snuke" pattern sequence.
start_state = (0, 0, 0) # (row, col, char_index)

# Add the starting state to the queue and mark it as visited.
q.append(start_state)
visited[0][0][0] = True

# Define possible movements from a cell: up, down, left, right.
# dr and dc arrays store the change in row and column indices respectively.
dr = [-1, 1, 0, 0] # Changes in row index
dc = [0, 0, -1, 1] # Changes in column index

# Flag to track if the target cell (H-1, W-1) has been reached.
found = False

# Start the BFS loop.
# The loop continues as long as there are states to explore in the queue.
while q:
    # Dequeue the next state to process: current cell (r, c) and the character index k it matched.
    r, c, k = q.popleft()

    # Check if the current cell (r, c) is the target destination (H-1, W-1).
    # Note: In 0-based indexing, (H, W) is (H-1, W-1).
    if r == H - 1 and c == W - 1:
        # If the target cell is reached, we have found a valid path.
        found = True
        break # Exit the BFS loop as the path is found.

    # Calculate the index of the character required for the next step in the path.
    # If the current cell (r, c) correctly matched target[k], the next adjacent cell
    # visited in the path must match the character at index (k+1) % 5 in the "snuke" sequence.
    next_k = (k + 1) % 5

    # Explore all four adjacent cells (neighbors) of the current cell (r, c).
    for i in range(4):
        # Calculate the coordinates of the neighboring cell.
        nr = r + dr[i]
        nc = c + dc[i]

        # Check if the neighboring cell (nr, nc) is within the grid boundaries [0, H-1] and [0, W-1].
        if 0 <= nr < H and 0 <= nc < W:
            # Check if the character at the neighboring cell matches the character required for the next step (next_k).
            if grid[nr][nc] == target[next_k]:
                # Check if this specific state (neighbor cell and the expected next character index)
                # has already been visited. Visiting the same cell with a different 'k'
                # might represent a different path or a different point in the "snuke" cycle,
                # so we only consider the state (nr, nc, next_k) visited if we reached it
                # previously having just matched target[next_k].
                if not visited[nr][nc][next_k]:
                    # If the state has not been visited, mark it as visited and add it to the queue
                    # for future exploration.
                    visited[nr][nc][next_k] = True
                    q.append((nr, nc, next_k))

# After the BFS loop finishes (either by finding the target or exhausting all reachable states),
# print "Yes" if the target was reached, otherwise print "No".
if found:
    print("Yes")
else:
    print("No")