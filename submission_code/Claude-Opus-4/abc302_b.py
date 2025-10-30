# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Define 8 directions: right, left, down, up, and 4 diagonals
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

target = "snuke"

# Try each starting position
for i in range(H):
    for j in range(W):
        # Try each direction
        for di, dj in directions:
            # Check if we can place all 5 letters in this direction
            valid = True
            positions = []
            
            for k in range(5):
                ni = i + k * di
                nj = j + k * dj
                
                # Check bounds
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    valid = False
                    break
                
                # Check if letter matches
                if grid[ni][nj] != target[k]:
                    valid = False
                    break
                
                positions.append((ni + 1, nj + 1))  # Convert to 1-indexed
            
            if valid:
                # Found the answer
                for pos in positions:
                    print(pos[0], pos[1])
                exit()