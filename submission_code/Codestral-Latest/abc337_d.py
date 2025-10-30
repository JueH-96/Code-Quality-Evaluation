# YOUR CODE HERE
import sys
from collections import deque

def min_operations_to_satisfy_condition(H, W, K, grid):
    # Function to check if there is a horizontal sequence of K 'o's
    def check_horizontal():
        for i in range(H):
            count = 0
            for j in range(W):
                if grid[i][j] == 'o':
                    count += 1
                    if count == K:
                        return True
                else:
                    count = 0
        return False

    # Function to check if there is a vertical sequence of K 'o's
    def check_vertical():
        for j in range(W):
            count = 0
            for i in range(H):
                if grid[i][j] == 'o':
                    count += 1
                    if count == K:
                        return True
                else:
                    count = 0
        return False

    # If the condition is already satisfied, return 0
    if check_horizontal() or check_vertical():
        return 0

    # BFS to find the minimum number of operations
    queue = deque()
    visited = set()

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                queue.append((i, j, 1))
                visited.add((i, j))

    while queue:
        x, y, ops = queue.popleft()

        # Change the current cell to 'o'
        grid[x][y] = 'o'

        # Check if the condition is satisfied
        if check_horizontal() or check_vertical():
            return ops

        # Add adjacent cells to the queue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.' and (nx, ny) not in visited:
                queue.append((nx, ny, ops + 1))
                visited.add((nx, ny))

    return -1

# Read input
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Calculate and print the result
result = min_operations_to_satisfy_condition(H, W, K, grid)
print(result)