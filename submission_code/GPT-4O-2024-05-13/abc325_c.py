# YOUR CODE HERE
def find_number_of_sensors(H, W, grid):
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                    grid[nx][ny] = '.'
                    stack.append((nx, ny))

    grid = [list(row) for row in grid]
    sensor_count = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                sensor_count += 1
                grid[i][j] = '.'
                dfs(i, j)

    return sensor_count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    print(find_number_of_sensors(H, W, grid))