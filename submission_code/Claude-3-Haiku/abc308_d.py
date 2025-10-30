from collections import deque

def can_reach_snuke(H, W, grid):
    # Define the target sequence
    target = 'snuke'

    # Initialize the queue and visited set
    queue = deque([(0, 0, 0)])  # (row, col, index)
    visited = set()

    while queue:
        row, col, index = queue.popleft()

        # Check if we have reached the target
        if row == H - 1 and col == W - 1 and index == len(target):
            return "Yes"

        # Check if the current cell matches the target character
        if grid[row][col] == target[index % len(target)]:
            index += 1

        # Add the adjacent cells to the queue
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and (new_row, new_col, index) not in visited:
                visited.add((new_row, new_col, index))
                queue.append((new_row, new_col, index))

    return "No"

# Read the input
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Solve the problem
print(can_reach_snuke(H, W, grid))