def count_paths(H, W, K, grid):
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Function to check if a position is valid
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'

    # DFS function to count paths
    def dfs(x, y, steps, visited):
        if steps == K:
            return 1  # Found a valid path of length K
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and (nx, ny) not in visited:
                visited.add((nx, ny))
                count += dfs(nx, ny, steps + 1, visited)
                visited.remove((nx, ny))
        return count

    total_paths = 0
    # Start DFS from each empty cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = set()
                visited.add((i, j))
                total_paths += dfs(i, j, 0, visited)

    return total_paths

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, K = map(int, data[0].split())
grid = [list(data[i + 1]) for i in range(H)]

# Calculate the number of paths
result = count_paths(H, W, K, grid)

# Print the result
print(result)