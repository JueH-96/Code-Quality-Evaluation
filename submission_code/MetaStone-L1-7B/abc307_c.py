# Read input
def read_sheet(h):
    sheet = []
    for _ in range(h):
        row = input().strip()
        sheet.append(row)
    return sheet

# Compute black squares for a given sheet
def get_black_squares(sheet):
    h = len(sheet)
    if h == 0:
        return []
    w = len(sheet[0])
    black = []
    for i in range(h):
        for j in range(w):
            if sheet[i][j] == '#':
                black.append((i, j))
    return black

# Read A, B, X
A = read_sheet(int(input()))
B = read_sheet(int(input()))
X = read_sheet(int(input()))

# Get black squares
A_black = get_black_squares(A)
B_black = get_black_squares(B)
X_black = get_black_squares(X)

# Compute minimal bounding box for A and B
all_black = A_black + B_black
if not all_black:
    print("No")
    exit()

min_row = min(i for (i, j) in all_black)
max_row = max(i for (i, j) in all_black)
min_col = min(j for (i, j) in all_black)
max_col = max(j for (i, j) in all_black)

h_A = len(A)
w_A = len(A[0]) if h_A > 0 else 0
h_B = len(B)
w_B = len(B[0]) if h_B > 0 else 0
h_X = len(X)
w_X = len(X[0]) if h_X > 0 else 0

# Check if minimal bounding box exceeds X's dimensions
if (max_row - min_row + 1) > h_X or (max_col - min_col + 1) > w_X:
    print("No")
    exit()

# Iterate all possible placements of A and B within X's window
found = False
for dx_A in range(h_X - h_A + 1):
    for dy_A in range(w_X - w_A + 1):
        for dx_B in range(h_X - h_B + 1):
            for dy_B in range(w_X - w_B + 1):
                # Create a grid filled with '.'
                grid = [['.' for _ in range(w_X)] for _ in range(h_X)]
                # Place A's black squares
                valid = True
                for (i, j) in A_black:
                    a_row = dx_A + i
                    a_col = dy_A + j
                    if a_row >= h_X or a_col >= w_X:
                        valid = False
                        break
                    grid[a_row][a_col] = '#'
                if not valid:
                    continue
                # Place B's black squares
                for (i, j) in B_black:
                    b_row = dx_B + i
                    b_col = dy_B + j
                    if b_row >= h_X or b_col >= w_X:
                        valid = False
                        break
                    grid[b_row][b_col] = '#'
                if not valid:
                    continue
                # Check if grid matches X
                match = True
                for i in range(h_X):
                    if not match:
                        break
                    for j in range(w_X):
                        if grid[i][j] != X[i][j]:
                            match = False
                            break
                if match:
                    print("Yes")
                    exit()

print("No")