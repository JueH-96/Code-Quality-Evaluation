H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Define all 8 possible directions (dx, dy)
directions = [ (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0) ]

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for dx, dy in directions:
                valid = True
                letters = []
                for step in range(1, 5):
                    ni = i + step * dx
                    nj = j + step * dy
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        valid = False
                        break
                    letters.append(grid[ni][nj])
                if valid and letters == ['n', 'u', 'k', 'e']:
                    # Output the positions
                    for step in range(5):
                        r = i + step * dx
                        c = j + step * dy
                        print(r + 1, c + 1)
                    exit()