# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    HA = int(data[idx])
    WA = int(data[idx+1])
    idx += 2
    A = []
    for _ in range(HA):
        A.append(data[idx])
        idx += 1
    
    HB = int(data[idx])
    WB = int(data[idx+1])
    idx += 2
    B = []
    for _ in range(HB):
        B.append(data[idx])
        idx += 1
    
    HX = int(data[idx])
    WX = int(data[idx+1])
    idx += 2
    X = []
    for _ in range(HX):
        X.append(data[idx])
        idx += 1
    
    # Extract black squares positions for A, B, X
    def get_black_positions(grid):
        positions = set()
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '#':
                    positions.add((i, j))
        return positions
    
    A_black = get_black_positions(A)
    B_black = get_black_positions(B)
    X_black = get_black_positions(X)
    
    # Check if the total number of black squares in A and B matches X
    if len(A_black) + len(B_black) != len(X_black):
        print("No")
        return
    
    # Try all possible shifts for A and B
    # Since the sheets are small, we can brute-force all possible shifts
    # We need to find a shift for A and B such that when combined, they match X
    
    # Determine the possible shifts for A and B
    # The shift is the difference in position between the top-left corner of A/B and the top-left corner of X
    
    # First, find the bounding box of X
    X_min_row = min(i for i, j in X_black)
    X_max_row = max(i for i, j in X_black)
    X_min_col = min(j for i, j in X_black)
    X_max_col = max(j for i, j in X_black)
    
    # Now, for A and B, find their bounding boxes
    A_min_row = min(i for i, j in A_black)
    A_max_row = max(i for i, j in A_black)
    A_min_col = min(j for i, j in A_black)
    A_max_col = max(j for i, j in A_black)
    
    B_min_row = min(i for i, j in B_black)
    B_max_row = max(i for i, j in B_black)
    B_min_col = min(j for i, j in B_black)
    B_max_col = max(j for i, j in B_black)
    
    # Calculate the possible shifts for A and B
    # The shift is such that when A is shifted by (shift_A_row, shift_A_col), its black squares are within the X bounding box
    # Similarly for B
    
    # For A, the shift must be such that:
    # A_min_row + shift_A_row >= X_min_row
    # A_max_row + shift_A_row <= X_max_row
    # A_min_col + shift_A_col >= X_min_col
    # A_max_col + shift_A_col <= X_max_col
    
    # Similarly for B
    
    # So, the possible shifts for A are:
    # shift_A_row can be from X_min_row - A_max_row to X_max_row - A_min_row
    # shift_A_col can be from X_min_col - A_max_col to X_max_col - A_min_col
    
    # Similarly for B
    
    # Now, iterate over all possible shifts for A and B
    # For each combination, check if the union of shifted A and B matches X
    
    # Calculate the possible shifts for A
    shift_A_row_min = X_min_row - A_max_row
    shift_A_row_max = X_max_row - A_min_row
    shift_A_col_min = X_min_col - A_max_col
    shift_A_col_max = X_max_col - A_min_col
    
    # Calculate the possible shifts for B
    shift_B_row_min = X_min_row - B_max_row
    shift_B_row_max = X_max_row - B_min_row
    shift_B_col_min = X_min_col - B_max_col
    shift_B_col_max = X_max_col - B_min_col
    
    # Iterate over all possible shifts for A and B
    for shift_A_row in range(shift_A_row_min, shift_A_row_max + 1):
        for shift_A_col in range(shift_A_col_min, shift_A_col_max + 1):
            # Shift A's black squares
            shifted_A = set()
            for (i, j) in A_black:
                shifted_A.add((i + shift_A_row, j + shift_A_col))
            
            for shift_B_row in range(shift_B_row_min, shift_B_row_max + 1):
                for shift_B_col in range(shift_B_col_min, shift_B_col_max + 1):
                    # Shift B's black squares
                    shifted_B = set()
                    for (i, j) in B_black:
                        shifted_B.add((i + shift_B_row, j + shift_B_col))
                    
                    # Combine the shifted A and B
                    combined = shifted_A.union(shifted_B)
                    
                    # Check if combined matches X_black
                    if combined == X_black:
                        print("Yes")
                        return
    
    # If no combination works
    print("No")

if __name__ == "__main__":
    main()