# YOUR CODE HERE
def find_snuke(grid):
    H, W = len(grid), len(grid[0])
    directions = [
        (0, 1), (1, 0), (1, 1), (-1, 1),
        (0, -1), (-1, 0), (-1, -1), (1, -1)
    ]
    target = "snuke"

    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W

    def check_direction(x, y, dx, dy):
        for i in range(5):
            nx, ny = x + i*dx, y + i*dy
            if not is_valid(nx, ny) or grid[nx][ny] != target[i]:
                return False
        return True

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    if check_direction(i, j, dx, dy):
                        return [(i+k*dx+1, j+k*dy+1) for k in range(5)]

    return None

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

result = find_snuke(grid)

for row, col in result:
    print(row, col)