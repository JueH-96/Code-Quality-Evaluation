# YOUR CODE HERE
def find_snake(H, W, grid):
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W
    
    def check_snake(x, y, dx, dy):
        positions = []
        for i in range(5):
            if not is_valid(x, y) or grid[x][y] != 'snake'[i]:
                return None
            positions.append((x + 1, y + 1))
            x += dx
            y += dy
        return positions
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    result = check_snake(i, j, dx, dy)
                    if result:
                        return result

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

positions = find_snake(H, W, grid)
for x, y in positions:
    print(x, y)