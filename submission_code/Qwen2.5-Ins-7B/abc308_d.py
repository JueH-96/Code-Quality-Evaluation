# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

def is_valid(x, y):
    return 0 <= x < H and 0 <= y < W

def dfs(x, y, path):
    if x == H-1 and y == W-1:
        return True
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if is_valid(nx, ny) and grid[nx][ny] == path[(len(path) - 1) % 8 + 1]:
            if dfs(nx, ny, path + grid[nx][ny]):
                return True
    return False

if grid[0][0] == 's' and dfs(0, 0, 's'):
    print('Yes')
else:
    print('No')