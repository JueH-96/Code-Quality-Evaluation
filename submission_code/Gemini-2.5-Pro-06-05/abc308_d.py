import sys
from collections import deque

def solve():
    """
    Reads grid input, finds if a "snuke" path exists from (1,1) to (H,W),
    and prints "Yes" or "No".
    """
    # Read grid dimensions and the grid itself from standard input.
    try:
        H, W = map(int, sys.stdin.readline().split())
        S = [sys.stdin.readline().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handles cases with malformed or empty input.
        return

    # The required character sequence.
    target = "snuke"

    # The path must start at (1,1) (0-indexed as (0,0)), and the character must be 's'.
    if S[0][0] != target[0]:
        print("No")
        return

    # Initialize a queue for Breadth-First Search (BFS).
    # The state is a tuple: (row, col, index_in_target_string).
    q = deque([(0, 0, 0)])
    
    # A set to keep track of visited states to avoid cycles and redundant work.
    visited = set([(0, 0, 0)])

    # Define the four possible moves: up, down, left, right.
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Start the BFS traversal.
    while q:
        r, c, idx = q.popleft()

        # If we have reached the destination (H,W), a valid path is found.
        # (H,W) is 1-based, which corresponds to (H-1, W-1) in 0-based indexing.
        if r == H - 1 and c == W - 1:
            print("Yes")
            return

        # Determine the properties for the next step in the path.
        next_idx = (idx + 1) % 5
        next_char = target[next_idx]

        # Explore all four adjacent cells.
        for dr, dc in moves:
            nr, nc = r + dr, c + dc

            # Check if the new cell is within the grid boundaries.
            if 0 <= nr < H and 0 <= nc < W:
                # Check if the character in the new cell matches the required next character.
                if S[nr][nc] == next_char:
                    # Form the new state for the neighbor.
                    new_state = (nr, nc, next_idx)
                    # If this state hasn't been visited, add it to the queue and visited set.
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append(new_state)

    # If the BFS completes without finding the destination, no such path exists.
    print("No")

# Execute the main solution logic.
solve()