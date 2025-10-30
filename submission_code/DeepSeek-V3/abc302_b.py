# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Define the target sequence
target = ['s', 'n', 'u', 'k', 'e']

# Directions: 8 possible directions (horizontal, vertical, diagonal)
directions = [
    (1, 0),   # down
    (-1, 0),  # up
    (0, 1),   # right
    (0, -1),  # left
    (1, 1),   # down-right
    (1, -1),  # down-left
    (-1, 1),  # up-right
    (-1, -1)  # up-left
]

# Iterate over each cell in the grid
for i in range(H):
    for j in range(W):
        if grid[i][j] == target[0]:
            # Try all directions
            for dx, dy in directions:
                x, y = i, j
                sequence = [(x+1, y+1)]  # 1-based indexing
                valid = True
                for k in range(1, 5):
                    x += dx
                    y += dy
                    if 0 <= x < H and 0 <= y < W:
                        if grid[x][y] == target[k]:
                            sequence.append((x+1, y+1))
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                if valid:
                    for pos in sequence:
                        print(pos[0], pos[1])
                    exit()