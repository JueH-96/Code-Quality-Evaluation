import sys

# Read all input data
data = sys.stdin.read().split()
index = 0

# Read dimensions and grid for sheet A
H_A, W_A = int(data[index]), int(data[index + 1])
index += 2
A_grid = [data[index + i] for i in range(H_A)]
index += H_A

# Read dimensions and grid for sheet B
H_B, W_B = int(data[index]), int(data[index + 1])
index += 2
B_grid = [data[index + i] for i in range(H_B)]
index += H_B

# Read dimensions and grid for sheet X
H_X, W_X = int(data[index]), int(data[index + 1])
index += 2
X_grid = [data[index + i] for i in range(H_X)]

# Compute the bounding box for black squares in A
min_row_A = min(i for i in range(H_A) if any(c == '#' for c in A_grid[i]))
max_row_A = max(i for i in range(H_A) if any(c == '#' for c in A_grid[i]))
min_col_A = min(j for j in range(W_A) if any(A_grid[i][j] == '#' for i in range(H_A)))
max_col_A = max(j for j in range(W_A) if any(A_grid[i][j] == '#' for i in range(H_A)))

# Compute the bounding box for black squares in B
min_row_B = min(i for i in range(H_B) if any(c == '#' for c in B_grid[i]))
max_row_B = max(i for i in range(H_B) if any(c == '#' for c in B_grid[i]))
min_col_B = min(j for j in range(W_B) if any(B_grid[i][j] == '#' for i in range(H_B)))
max_col_B = max(j for j in range(W_B) if any(B_grid[i][j] == '#' for i in range(H_B)))

# Calculate the range of possible offsets for A and B relative to X
low_dx_a = -min_row_A
high_dx_a = H_X - 1 - max_row_A
low_dy_a = -min_col_A
high_dy_a = W_X - 1 - max_col_A

low_dx_b = -min_row_B
high_dx_b = H_X - 1 - max_row_B
low_dy_b = -min_col_B
high_dy_b = W_X - 1 - max_col_B

# Function to check if the union of A and B matches X for given offsets
def check(dx_a, dy_a, dx_b, dy_b):
    for r in range(H_X):
        for c in range(W_X):
            a_contrib = False
            if dx_a <= r <= dx_a + H_A - 1 and dy_a <= c <= dy_a + W_A - 1:
                if A_grid[r - dx_a][c - dy_a] == '#':
                    a_contrib = True
            b_contrib = False
            if dx_b <= r <= dx_b + H_B - 1 and dy_b <= c <= dy_b + W_B - 1:
                if B_grid[r - dx_b][c - dy_b] == '#':
                    b_contrib = True
            black_value = a_contrib or b_contrib
            if (black_value and X_grid[r][c] != '#') or (not black_value and X_grid[r][c] == '#'):
                return False
    return True

# Iterate through all possible offset combinations
for dx_a_val in range(low_dx_a, high_dx_a + 1):
    for dy_a_val in range(low_dy_a, high_dy_a + 1):
        for dx_b_val in range(low_dx_b, high_dx_b + 1):
            for dy_b_val in range(low_dy_b, high_dy_b + 1):
                if check(dx_a_val, dy_a_val, dx_b_val, dy_b_val):
                    print("Yes")
                    sys.exit()

# If no valid placement found
print("No")