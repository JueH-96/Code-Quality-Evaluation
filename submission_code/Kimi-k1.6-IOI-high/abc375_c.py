n = int(input())
grid = [list(input().strip()) for _ in range(n)]

for l in range(n // 2):
    for y in range(l, n - l - 1):
        x = l
        pos1_x, pos1_y = x, y
        pos2_x, pos2_y = y, (n - 1) - x
        pos3_x, pos3_y = (n - 1) - x, (n - 1) - y
        pos4_x, pos4_y = (n - 1) - y, x
        
        a = grid[pos1_x][pos1_y]
        b = grid[pos2_x][pos2_y]
        c = grid[pos3_x][pos3_y]
        d = grid[pos4_x][pos4_y]
        
        grid[pos2_x][pos2_y] = a
        grid[pos3_x][pos3_y] = b
        grid[pos4_x][pos4_y] = c
        grid[pos1_x][pos1_y] = d

for row in grid:
    print(''.join(row))