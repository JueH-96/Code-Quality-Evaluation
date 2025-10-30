H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Define eight possible directions
directions = [
    (0, 1),   # Right
    (0, -1),  # Left
    (1, 0),   # Down
    (-1, 0),  # Up
    (1, 1),   # Down-Right
    (1, -1),  # Down-Left
    (-1, 1),  # Up-Right
    (-1, -1)  # Up-Left
]

# The target sequence
target = ['s', 'n', 'u', 'k', 'e']

for i in range(H):
    for j in range(W):
        if grid[i][j] == 's':
            for dx, dy in directions:
                positions = []
                x, y = i, j
                for k in range(5):
                    nx, ny = x + dx * k, y + dy * k
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] != target[k]:
                            break
                        positions.append((nx + 1, ny + 1))  # 1-based indexing
                    else:
                        break
                else:
                    # If all positions are valid and match the target
                    for pos in positions:
                        print(pos[0], pos[1])
                    exit()