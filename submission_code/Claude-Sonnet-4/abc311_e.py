# YOUR CODE HERE
H, W, N = map(int, input().split())

# Create a grid to mark holed squares
holed = [[False] * (W + 1) for _ in range(H + 1)]

for _ in range(N):
    a, b = map(int, input().split())
    holed[a][b] = True

# Create 2D prefix sum array for holes
prefix = [[0] * (W + 1) for _ in range(H + 1)]

for i in range(1, H + 1):
    for j in range(1, W + 1):
        prefix[i][j] = prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1] + (1 if holed[i][j] else 0)

def count_holes_in_rectangle(r1, c1, r2, c2):
    # Count holes in rectangle from (r1,c1) to (r2,c2) inclusive
    return prefix[r2][c2] - prefix[r1-1][c2] - prefix[r2][c1-1] + prefix[r1-1][c1-1]

count = 0

# Try all possible squares
for i in range(1, H + 1):
    for j in range(1, W + 1):
        # Try all possible sizes starting from (i,j)
        max_size = min(H - i + 1, W - j + 1)
        for n in range(1, max_size + 1):
            # Check if n×n square starting at (i,j) has no holes
            if count_holes_in_rectangle(i, j, i + n - 1, j + n - 1) == 0:
                count += 1
            else:
                # If current size has holes, larger sizes will also have holes
                break

print(count)