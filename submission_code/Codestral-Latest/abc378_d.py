from collections import deque

def count_paths(H, W, K, grid):
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'

    def bfs(start):
        queue = deque([(start[0], start[1], 0, {start})])
        count = 0

        while queue:
            x, y, moves, visited = queue.popleft()

            if moves == K:
                count += 1
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and (nx, ny) not in visited:
                    queue.append((nx, ny, moves + 1, visited | {(nx, ny)}))

        return count

    total_paths = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_paths += bfs((i, j))

    return total_paths

# Read input
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Calculate and print the result
result = count_paths(H, W, K, grid)
print(result)