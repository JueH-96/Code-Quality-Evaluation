import sys

def read_sheet(H, W):
    """Reads a sheet from standard input."""
    sheet = [sys.stdin.readline().strip() for _ in range(H)]
    return sheet

def find_black_bounds(sheet, H, W):
    """Finds the minimum and maximum row/column indices containing a '#'."""
    min_r, max_r, min_c, max_c = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if sheet[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    # Problem constraint guarantees at least one black square, so max_r != -1
    return min_r, max_r, min_c, max_c

def is_sheet_black_at_offset(r_combined, c_combined, r_offset, c_offset, sheet, H_sheet, W_sheet):
    """
    Checks if the cell (r_combined, c_combined) in the combined grid is black
    due to the given sheet placed with its top-left at (r_offset, c_offset).
    """
    r_sheet = r_combined - r_offset
    c_sheet = c_combined - c_offset
    # Check if the coordinate (r_sheet, c_sheet) is within the sheet's bounds
    if 0 <= r_sheet < H_sheet and 0 <= c_sheet < W_sheet:
        return sheet[r_sheet][c_sheet] == '#'
    return False

def check_placement(r_A, c_A, sheet_A, H_A, W_A, r_B, c_B, sheet_B, H_B, W_B, sheet_X, H_X, W_X):
    """
    Checks if placing sheet A at (r_A, c_A) and sheet B at (r_B, c_B)
    relative to X's top-left results in the combined shape matching X.
    This function also implicitly checks that all black squares fall within X's bounds,
    because the calling loops iterate over valid (r_A, c_A, r_B, c_B) ranges
    derived to satisfy the "all black squares included" condition.
    """
    for r_x in range(H_X):
        for c_x in range(W_X):
            # Check if the cell (r_x, c_x) in the combined sheet should be black
            is_combined_black = is_sheet_black_at_offset(r_x, c_x, r_A, c_A, sheet_A, H_A, W_A) or \
                                is_sheet_black_at_offset(r_x, c_x, r_B, c_B, sheet_B, H_B, W_B)

            # Compare with the required state in sheet X
            if (sheet_X[r_x][c_x] == '#') != is_combined_black:
                 return False # Mismatch

    # If all cells in the X area match, this placement works.
    return True

# Read inputs
H_A, W_A = map(int, sys.stdin.readline().split())
sheet_A = read_sheet(H_A, W_A)
H_B, W_B = map(int, sys.stdin.readline().split())
sheet_B = read_sheet(H_B, W_B)
H_X, W_X = map(int, sys.stdin.readline().split())
sheet_X = read_sheet(H_X, W_X)

# Find black square bounds for A and B
# These bounds are relative to the sheet's own top-left (0,0)
min_ar, max_ar, min_ac, max_ac = find_black_bounds(sheet_A, H_A, W_A)
min_br, max_br, min_bc, max_bc = find_black_bounds(sheet_B, H_B, W_B)

# Iterate through possible top-left positions (r_A, c_A) of sheet A relative to X's top-left (0,0).
# The range for r_A is determined by the requirement that all black rows [min_ar, max_ar] of A,
# when shifted by r_A, must fall within X's rows [0, H_X-1].
# r_A + min_ar >= 0  => r_A >= -min_ar
# r_A + max_ar < H_X => r_A < H_X - max_ar
# So r_A ranges from -min_ar up to H_X - max_ar - 1 (inclusive).
for r_A in range(-min_ar, H_X - max_ar):
    for c_A in range(-min_ac, W_X - max_ac):
        # Iterate through possible top-left positions (r_B, c_B) of sheet B relative to X's top-left (0,0).
        # Similar logic for r_B and c_B.
        for r_B in range(-min_br, H_X - max_br):
            for c_B in range(-min_bc, W_X - max_bc):
                # Check if this placement combination results in sheet X
                if check_placement(r_A, c_A, sheet_A, H_A, W_A, r_B, c_B, sheet_B, H_B, W_B, sheet_X, H_X, W_X):
                    print("Yes")
                    sys.exit() # Found a solution, exit immediately

# If loops complete without finding a solution
print("No")