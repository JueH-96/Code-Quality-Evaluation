import sys

def find_positions(grid, H, W):
    directions = [
        (0, 1), (1, 0), (1, 1), (1, -1),
        (0, -1), (-1, 0), (-1, -1), (-1, 1)
    ]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W
    
    def check_line(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
        # Check if all points are on the same line
        if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1) and \
           (y3 - y1) * (x4 - x1) == (y4 - y1) * (x3 - x1) and \
           (y4 - y1) * (x5 - x1) == (y5 - y1) * (x4 - x1):
            return True
        return False
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    x1, y1 = i, j
                    x2, y2 = x1 + dx, y1 + dy
                    x3, y3 = x2 + dx, y2 + dy
                    x4, y4 = x3 + dx, y3 + dy
                    x5, y5 = x4 + dx, y4 + dy
                    if is_valid(x2, y2) and is_valid(x3, y3) and is_valid(x4, y4) and is_valid(x5, y5):
                        if grid[x2][y2] == 'n' and grid[x3][y3] == 'u' and grid[x4][y4] == 'k' and grid[x5][y5] == 'e':
                            if check_line(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5):
                                return [(x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (x3 + 1, y3 + 1), (x4 + 1, y4 + 1), (x5 + 1, y5 + 1)]
    return []

input_lines = sys.stdin.read().strip().split('
')
H, W = map(int, input_lines[0].split())
grid = input_lines[1:]

positions = find_positions(grid, H, W)
for pos in positions:
    print(f"{pos[0]} {pos[1]}")