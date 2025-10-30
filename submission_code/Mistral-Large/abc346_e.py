import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
M = int(data[2])

operations = []
for i in range(M):
    T = int(data[3 + i * 3])
    A = int(data[4 + i * 3])
    X = int(data[5 + i * 3])
    operations.append((T, A, X))

# Initialize the grid with color 0
grid = [[0] * W for _ in range(H)]

# Perform the operations
for op in operations:
    T, A, X = op
    if T == 1:
        # Repaint all cells in the A-th row with color X
        for j in range(W):
            grid[A-1][j] = X
    elif T == 2:
        # Repaint all cells in the A-th column with color X
        for i in range(H):
            grid[i][A-1] = X

# Count the number of cells for each color
color_count = {}
for row in grid:
    for color in row:
        if color in color_count:
            color_count[color] += 1
        else:
            color_count[color] = 1

# Prepare the output
K = len(color_count)
output = [f"{K}"]
for color in sorted(color_count.keys()):
    output.append(f"{color} {color_count[color]}")

# Print the output
print("
".join(output))