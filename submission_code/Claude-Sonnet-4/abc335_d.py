N = int(input())

# Initialize grid
grid = [[0 for _ in range(N)] for _ in range(N)]

# Place Takahashi at center
center = N // 2
grid[center][center] = 'T'

# Create path using a simple pattern that avoids center
# We'll go row by row, alternating direction, and skip center
path = []

for i in range(N):
    if i % 2 == 0:  # Even rows: left to right
        for j in range(N):
            if i != center or j != center:  # Skip center
                path.append((i, j))
    else:  # Odd rows: right to left
        for j in range(N-1, -1, -1):
            if i != center or j != center:  # Skip center
                path.append((i, j))

# Place dragon parts
for idx, (i, j) in enumerate(path):
    grid[i][j] = idx + 1

# Print result
for i in range(N):
    print(' '.join(str(grid[i][j]) for j in range(N)))