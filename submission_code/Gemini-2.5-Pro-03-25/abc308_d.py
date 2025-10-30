# YOUR CODE HERE
import sys
import collections

# sys.setrecursionlimit(2000000) # Increase recursion depth if using DFS, not necessary for BFS

def solve():
    # Read grid dimensions H (rows) and W (columns) from standard input
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid content. S is a list of H strings, each of length W.
    # S[i] represents the i-th row (0-indexed) of the grid.
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # The target sequence of characters that the path must follow.
    target = "snuke"
    
    # The problem requires the path to start at cell (1,1).
    # In 0-based indexing used internally, this corresponds to cell (0,0).
    # The first character on the path must be 's', which is target[0].
    # Check if the character at the starting cell (0,0) matches 's'.
    # If not, a valid path according to the rules cannot start, so print "No" and exit.
    if S[0][0] != target[0]:
        print("No")
        return

    # Keep track of visited states to avoid cycles and redundant computations.
    # A state is defined by a tuple (row, column, index_in_target_sequence).
    # `visited[r][c][idx]` is True if the state (r, c, idx) has been reached.
    # The dimensions are H rows, W columns, and 5 possible indices in the target sequence.
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]

    # Initialize a queue for Breadth-First Search (BFS).
    # Using `collections.deque` provides efficient append (enqueue) and popleft (dequeue) operations.
    # The queue stores states as tuples: (row, col, target_idx).
    q = collections.deque()

    # Add the initial state to the queue.
    # The path starts at cell (0,0). This cell corresponds to the 0-th character ('s') of the target sequence.
    # The state is thus (0, 0, 0).
    q.append((0, 0, 0))
    # Mark the initial state as visited.
    visited[0][0][0] = True

    # Perform BFS while the queue is not empty.
    while q:
        # Dequeue the next state (current cell coordinates and sequence index) to process.
        r, c, target_idx = q.popleft()

        # Check if the current cell (r, c) is the destination cell.
        # The destination cell is (H,W) in 1-based indexing, which is (H-1, W-1) in 0-based indexing.
        if r == H - 1 and c == W - 1:
            # If the destination is reached, a valid path exists. Print "Yes" and exit.
            print("Yes")
            return

        # Determine the index of the *next* character required in the sequence.
        # The sequence "snuke" has length 5, so indices wrap around using modulo 5.
        next_target_idx = (target_idx + 1) % 5
        # Get the actual character expected at the next cell in the path.
        expected_char = target[next_target_idx]

        # Explore adjacent cells (neighbors): up, down, left, right.
        # `dr, dc` represent the change in row and column indices for each direction.
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Calculate the coordinates of the neighbor cell.
            nr, nc = r + dr, c + dc

            # Check if the neighbor cell (nr, nc) is within the grid boundaries.
            # Row index `nr` must be in [0, H-1]. Column index `nc` must be in [0, W-1].
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the character in the neighbor cell matches the expected character for the next step in the sequence.
                if S[nr][nc] == expected_char:
                    # Check if the state represented by (nr, nc, next_target_idx) has already been visited.
                    # This check is crucial to prevent infinite loops in case of cycles in the grid path 
                    # and to avoid redundant exploration of states.
                    if not visited[nr][nc][next_target_idx]:
                        # If this state has not been visited, mark it as visited.
                        visited[nr][nc][next_target_idx] = True
                        # Enqueue this new state (neighbor cell coordinates and corresponding sequence index) for further exploration.
                        q.append((nr, nc, next_target_idx))

    # If the queue becomes empty, it means BFS has explored all reachable states starting from (0,0) 
    # following the "snuke" sequence constraint, and none of the paths reached the destination (H-1, W-1).
    # Therefore, no valid path exists that satisfies all conditions. Print "No".
    print("No")

# Call the solve function to execute the program logic using standard input/output.
solve()