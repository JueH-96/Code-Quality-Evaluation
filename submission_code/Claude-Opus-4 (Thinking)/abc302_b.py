# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

target = "snuke"

# Define 8 directions: (dr, dc)
directions = [
    (0, 1),   # right
    (0, -1),  # left
    (1, 0),   # down
    (-1, 0),  # up
    (1, 1),   # down-right
    (1, -1),  # down-left
    (-1, 1),  # up-right
    (-1, -1)  # up-left
]

found = False
for i in range(H):
    if found:
        break
    for j in range(W):
        if found:
            break
        if grid[i][j] == 's':
            # Try all 8 directions
            for dr, dc in directions:
                positions = []
                valid = True
                
                for k in range(5):
                    ni = i + k * dr
                    nj = j + k * dc
                    
                    # Check bounds
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        valid = False
                        break
                    
                    # Check character
                    if grid[ni][nj] != target[k]:
                        valid = False
                        break
                    
                    positions.append((ni + 1, nj + 1))  # 1-indexed
                
                if valid:
                    for r, c in positions:
                        print(r, c)
                    found = True
                    break