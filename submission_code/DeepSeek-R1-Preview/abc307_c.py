# Read input for sheet A
H_A, W_A = map(int, input().split())
A = [input().strip() for _ in range(H_A)]
A_black = [(i, j) for i in range(H_A) for j in range(W_A) if A[i][j] == '#']

# Read input for sheet B
H_B, W_B = map(int, input().split())
B = [input().strip() for _ in range(H_B)]
B_black = [(i, j) for i in range(H_B) for j in range(W_B) if B[i][j] == '#']

# Read input for sheet X
H_X, W_X = map(int, input().split())
X = [input().strip() for _ in range(H_X)]

# Function to compute possible offsets
def compute_offsets(black_squares, h_x, w_x):
    if not black_squares:
        return []
    # Compute a_min and a_max
    i_list = [i for (i, j) in black_squares]
    a_min = max(-i for i in i_list)
    a_max = min((h_x - 1) - i for i in i_list)
    if a_min > a_max:
        return []
    # Compute b_min and b_max
    j_list = [j for (i, j) in black_squares]
    b_min = max(-j for j in j_list)
    b_max = min((w_x - 1) - j for j in j_list)
    if b_min > b_max:
        return []
    # Generate all (a, b) pairs
    offsets = []
    for a in range(a_min, a_max + 1):
        for b in range(b_min, b_max + 1):
            offsets.append((a, b))
    return offsets

# Compute possible offsets for A and B
A_offsets = compute_offsets(A_black, H_X, W_X)
B_offsets = compute_offsets(B_black, H_X, W_X)

# Check if either A or B cannot be placed
if not A_offsets or not B_offsets:
    print("No")
    exit()

# Check all combinations of A and B positions
for a, b in A_offsets:
    for c, d in B_offsets:
        # Initialize grid
        grid = [['.' for _ in range(W_X)] for _ in range(H_X)]
        # Place A
        for i, j in A_black:
            x = a + i
            y = b + j
            grid[x][y] = '#'
        # Place B
        for i, j in B_black:
            x = c + i
            y = d + j
            grid[x][y] = '#'
        # Convert grid to strings
        current = [''.join(row) for row in grid]
        if current == X:
            print("Yes")
            exit()

# If no combination matches
print("No")