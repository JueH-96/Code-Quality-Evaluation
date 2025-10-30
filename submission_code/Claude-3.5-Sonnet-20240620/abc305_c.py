# YOUR CODE HERE
H, W = map(int, input().split())
grid = [input() for _ in range(H)]

# Find the rectangle boundaries
left = W
right = 0
top = H
bottom = 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            left = min(left, j)
            right = max(right, j)
            top = min(top, i)
            bottom = max(bottom, i)

# Check each position in the rectangle
for i in range(top, bottom + 1):
    for j in range(left, right + 1):
        # If it's a dot and all its neighbors (including diagonals) are '#', this is the eaten cookie
        if grid[i][j] == '.':
            neighbors = [
                (i-1, j-1), (i-1, j), (i-1, j+1),
                (i, j-1), (i, j+1),
                (i+1, j-1), (i+1, j), (i+1, j+1)
            ]
            if all(0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#' for ni, nj in neighbors):
                print(i+1, j+1)
                exit()

# If we haven't found the eaten cookie inside, it must be on the edge
for i in range(top, bottom + 1):
    if grid[i][left] == '.':
        print(i+1, left+1)
        exit()
    if grid[i][right] == '.':
        print(i+1, right+1)
        exit()

for j in range(left, right + 1):
    if grid[top][j] == '.':
        print(top+1, j+1)
        exit()
    if grid[bottom][j] == '.':
        print(bottom+1, j+1)
        exit()