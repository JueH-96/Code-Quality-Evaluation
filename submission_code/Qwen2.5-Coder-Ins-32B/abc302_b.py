# YOUR CODE HERE
def find_snuke(grid, H, W):
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 's':
                for dx, dy in directions:
                    found = True
                    positions = [(i, j)]
                    x, y = i, j
                    for k in range(1, 5):
                        nx, ny = x + k * dx, y + k * dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == 'snuke'[k]:
                            positions.append((nx, ny))
                        else:
                            found = False
                            break
                    if found:
                        return positions

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

positions = find_snuke(grid, H, W)
for pos in positions:
    print(pos[0] + 1, pos[1] + 1)