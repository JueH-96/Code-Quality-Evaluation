def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    HA, WA = int(data[idx]), int(data[idx+1])
    idx += 2
    A = [data[idx+i] for i in range(HA)]
    idx += HA
    
    HB, WB = int(data[idx]), int(data[idx+1])
    idx += 2
    B = [data[idx+i] for i in range(HB)]
    idx += HB
    
    HX, WX = int(data[idx]), int(data[idx+1])
    idx += 2
    X = [data[idx+i] for i in range(HX)]
    
    # Extract black squares positions for A, B, X
    def get_black_positions(grid):
        positions = []
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '#':
                    positions.append((i, j))
        return positions
    
    A_black = get_black_positions(A)
    B_black = get_black_positions(B)
    X_black = get_black_positions(X)
    
    # Check if the number of black squares matches
    if len(A_black) + len(B_black) != len(X_black):
        print("No")
        return
    
    # Try all possible shifts for A and B
    # Since the sheets are small, we can brute-force
    # We need to find a shift for A and B such that when combined, they match X
    
    # First, find the bounding box for X
    min_x_row = min([x[0] for x in X_black])
    max_x_row = max([x[0] for x in X_black])
    min_x_col = min([x[1] for x in X_black])
    max_x_col = max([x[1] for x in X_black])
    
    # Now, try all possible shifts for A and B
    # The shift is the difference in position when pasting A and B onto C
    # We need to find shifts such that all black squares of A and B are within the X bounding box
    # and their combined positions match X_black
    
    # Since the sheets are small, we can try all possible shifts within a reasonable range
    # For A, the shift can be from -HA to HX
    # Similarly for B
    
    # To limit the shifts, we can consider the relative positions of the black squares
    # For A, the shift must be such that the black squares are within the X bounding box
    # So, for each black square in A, the shifted position must be within [min_x_row, max_x_row] and [min_x_col, max_x_col]
    
    # Similarly for B
    
    # So, for A, the shift in row must be such that:
    # min_x_row <= a_row + shift_a_row <= max_x_row
    # Similarly for columns
    
    # So, for each black square in A, we can compute the possible shift ranges
    # Then, the overall shift for A must be within the intersection of all these ranges
    
    # Similarly for B
    
    # Let's compute the possible shift ranges for A and B
    
    # For A
    shift_a_row_min = -HA
    shift_a_row_max = HX
    shift_a_col_min = -WA
    shift_a_col_max = WX
    
    # For B
    shift_b_row_min = -HB
    shift_b_row_max = HX
    shift_b_col_min = -WB
    shift_b_col_max = WX
    
    # Now, for each possible shift for A and B, check if the combined black squares match X_black
    
    # Since the sheets are small, we can iterate over all possible shifts
    # For A, the shift in row can be from shift_a_row_min to shift_a_row_max
    # Similarly for columns
    
    # For B, the same
    
    # So, we have a nested loop over all possible shifts for A and B
    
    # To optimize, we can limit the shifts based on the black squares positions
    
    # For A, the shift must be such that all black squares of A are within the X bounding box
    # So, for each black square in A, the shifted position must be within [min_x_row, max_x_row] and [min_x_col, max_x_col]
    
    # So, for each black square in A, we can compute the possible shift ranges
    # Then, the overall shift for A must be within the intersection of all these ranges
    
    # Similarly for B
    
    # Let's compute the possible shift ranges for A and B based on their black squares
    
    # For A
    a_row_shifts = set()
    a_col_shifts = set()
    for a_row, a_col in A_black:
        # The shifted row must be in [min_x_row, max_x_row]
        # So, shift_a_row must be in [min_x_row - a_row, max_x_row - a_row]
        a_row_shifts.add((min_x_row - a_row, max_x_row - a_row))
        # Similarly for columns
        a_col_shifts.add((min_x_col - a_col, max_x_col - a_col))
    
    # Now, find the intersection of all row shifts and column shifts for A
    a_row_min = max([x[0] for x in a_row_shifts])
    a_row_max = min([x[1] for x in a_row_shifts])
    a_col_min = max([x[0] for x in a_col_shifts])
    a_col_max = min([x[1] for x in a_col_shifts])
    
    # Similarly for B
    b_row_shifts = set()
    b_col_shifts = set()
    for b_row, b_col in B_black:
        b_row_shifts.add((min_x_row - b_row, max_x_row - b_row))
        b_col_shifts.add((min_x_col - b_col, max_x_col - b_col))
    
    b_row_min = max([x[0] for x in b_row_shifts])
    b_row_max = min([x[1] for x in b_row_shifts])
    b_col_min = max([x[0] for x in b_col_shifts])
    b_col_max = min([x[1] for x in b_col_shifts])
    
    # Now, iterate over all possible shifts for A and B within these ranges
    for shift_a_row in range(a_row_min, a_row_max + 1):
        for shift_a_col in range(a_col_min, a_col_max + 1):
            for shift_b_row in range(b_row_min, b_row_max + 1):
                for shift_b_col in range(b_col_min, b_col_max + 1):
                    # Compute the positions of all black squares after shifting
                    a_shifted = [(a_row + shift_a_row, a_col + shift_a_col) for a_row, a_col in A_black]
                    b_shifted = [(b_row + shift_b_row, b_col + shift_b_col) for b_row, b_col in B_black]
                    # Combine the positions
                    combined = set(a_shifted + b_shifted)
                    # Check if the combined positions match X_black
                    if combined == set(X_black):
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()