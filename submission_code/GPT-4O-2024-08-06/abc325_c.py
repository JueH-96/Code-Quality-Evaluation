# YOUR CODE HERE
def count_sensor_groups(H, W, grid):
    # Directions for 8 possible moves (horizontal, vertical, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    # Visited matrix to keep track of visited sensors
    visited = [[False] * W for _ in range(H)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '#' and not visited[x][y]
    
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny):
                    visited[nx][ny] = True
                    stack.append((nx, ny))
    
    sensor_groups = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                # Found an unvisited sensor, start a new DFS
                visited[i][j] = True
                dfs(i, j)
                sensor_groups += 1
    
    return sensor_groups

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = data[1:H+1]
    
    result = count_sensor_groups(H, W, grid)
    print(result)