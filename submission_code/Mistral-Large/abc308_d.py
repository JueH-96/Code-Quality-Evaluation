import sys
from collections import deque

input = sys.stdin.read

def can_reach_destination(H, W, grid):
    # Directions for moving in the grid (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # The sequence of characters that must be followed
    sequence = "snuke"
    # The length of the sequence
    seq_len = len(sequence)

    # Queue for BFS
    queue = deque([(0, 0, 0)])  # (row, column, sequence index)
    # Visited set to keep track of visited cells and their sequence index
    visited = set((0, 0, 0))

    while queue:
        r, c, seq_idx = queue.popleft()

        # If we reach the bottom-right corner with the correct sequence index
        if (r, c) == (H-1, W-1) and seq_idx % seq_len == 4:
            return "Yes"

        # Explore all possible directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            new_seq_idx = (seq_idx + 1) % seq_len

            # Check if the new position is within bounds and matches the sequence character
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == sequence[new_seq_idx]:
                if (nr, nc, new_seq_idx) not in visited:
                    visited.add((nr, nc, new_seq_idx))
                    queue.append((nr, nc, new_seq_idx))

    return "No"

# Read input
data = input().split()
H = int(data[0])
W = int(data[1])
grid = [data[i] for i in range(2, H+2)]

# Output the result
print(can_reach_destination(H, W, grid))