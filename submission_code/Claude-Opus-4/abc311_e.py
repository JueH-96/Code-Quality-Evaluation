# YOUR CODE HERE
H, W, N = map(int, input().split())

# Create a grid to mark holed cells
holed = [[0] * (W + 1) for _ in range(H + 1)]

# Read holed positions and mark them (1-indexed)
for _ in range(N):
    a, b = map(int, input().split())
    holed[a][b] = 1

# Create prefix sum array for quick range queries
prefix = [[0] * (W + 1) for _ in range(H + 1)]
for i in range(1, H + 1):
    for j in range(1, W + 1):
        prefix[i][j] = holed[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]

# Function to check if a square region has any holes
def has_hole(r1, c1, r2, c2):
    # Returns the sum of holed cells in rectangle from (r1,c1) to (r2,c2)
    return prefix[r2][c2] - prefix[r1-1][c2] - prefix[r2][c1-1] + prefix[r1-1][c1-1]

count = 0

# Try all possible top-left corners
for i in range(1, H + 1):
    for j in range(1, W + 1):
        # For each top-left corner, try all possible sizes
        max_size = min(H - i + 1, W - j + 1)
        
        # Binary search or linear search for maximum valid size
        for n in range(1, max_size + 1):
            # Check if square from (i,j) to (i+n-1, j+n-1) has any holes
            if has_hole(i, j, i + n - 1, j + n - 1) > 0:
                break
            count += 1

print(count)