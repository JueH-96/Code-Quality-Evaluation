import sys
from collections import deque

def count_paths(H, W, K, grid):
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS to count the number of valid paths
    def bfs(start):
        queue = deque([(start, 0)])  # (current_position, steps_taken)
        visited = set()
        visited.add(start)
        path_count = 0

        while queue:
            (current_row, current_col), steps = queue.popleft()

            if steps == K:
                path_count += 1
                continue

            for dr, dc in directions:
                new_row, new_col = current_row + dr, current_col + dc
                if 1 <= new_row <= H and 1 <= new_col <= W and grid[new_row-1][new_col-1] == '.' and (new_row, new_col) not in visited:
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), steps + 1))

        return path_count

    total_paths = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_paths += bfs((i+1, j+1))

    return total_paths

# Read input
input = sys.stdin.read
data = input().split()
H = int(data[0])
W = int(data[1])
K = int(data[2])
grid = [data[3+i:3+i+W] for i in range(0, H*W, W)]

# Calculate and print the result
result = count_paths(H, W, K, grid)
print(result)