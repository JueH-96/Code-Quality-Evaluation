import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1

# Read the grid
grid = []
for i in range(H):
    s = data[index]
    index += 1
    grid.append(s)

# Define the target string "snuke"
target = "snuke"

# Define all possible directions (8 directions)
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Iterate over all cells in the grid
for r in range(H):  # r is row index, 0-based
    for c in range(W):  # c is column index, 0-based
        if grid[r][c] == 's':  # Start only from cells with 's'
            for dr, dc in directions:  # Try all 8 directions
                valid = True
                # Check all five positions in the path
                for k in range(5):
                    row_k = r + k * dr
                    col_k = c + k * dc
                    # Check bounds
                    if not (0 <= row_k < H and 0 <= col_k < W):
                        valid = False
                        break
                    # Check the letter matches the k-th character in "snuke"
                    if grid[row_k][col_k] != target[k]:
                        valid = False
                        break
                # If all positions are valid, this is the path
                if valid:
                    # Output the positions in 1-based indexing
                    for k in range(5):
                        row_out = r + k * dr + 1  # Convert to 1-based
                        col_out = c + k * dc + 1  # Convert to 1-based
                        print(f"{row_out} {col_out}")
                    # Exit after printing the result
                    sys.exit()