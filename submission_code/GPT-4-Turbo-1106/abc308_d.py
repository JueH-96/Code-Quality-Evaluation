from collections import deque

# Read input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Define the sequence of letters to be followed
sequence = "snuke"

# Define directions for adjacent cells
directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

# Initialize queue for BFS and visited set
queue = deque([(0, 0, 0)])  # (row, col, index in sequence)
visited = set()

# BFS to find the path
while queue:
    r, c, seq_idx = queue.popleft()
    # Check if we reached the end
    if (r, c) == (H - 1, W - 1) and grid[r][c] == sequence[seq_idx]:
        print("Yes")
        break
    # Check if the current cell matches the sequence
    if grid[r][c] != sequence[seq_idx]:
        continue
    # Mark as visited
    visited.add((r, c, seq_idx))
    # Move to adjacent cells
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc, (seq_idx + 1) % 5) not in visited:
            queue.append((nr, nc, (seq_idx + 1) % 5))
else:
    print("No")