def find_s_n_u_k_e(grid, H, W):
    target = "snuke"
    directions = [
        (0, 1),   # right
        (1, 0),   # down
        (1, 1),   # down-right diagonal
        (1, -1),  # down-left diagonal
        (0, -1),  # left
        (-1, 0),  # up
        (-1, -1), # up-left diagonal
        (-1, 1)   # up-right diagonal
    ]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 's':
                for dr, dc in directions:
                    positions = []
                    for i in range(5):
                        nr = r + dr * i
                        nc = c + dc * i
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == target[i]:
                            positions.append((nr + 1, nc + 1))  # Store 1-based index
                        else:
                            break
                    if len(positions) == 5:
                        # Check if they are on a common line at regular intervals
                        if all(positions[i][0] - positions[i + 1][0] == positions[0][0] - positions[1][0] for i in range(4)):
                            return positions
    return []

import sys
input = sys.stdin.read
data = input().strip().splitlines()

H, W = map(int, data[0].split())
grid = data[1:H + 1]

result = find_s_n_u_k_e(grid, H, W)
for r, c in result:
    print(r, c)