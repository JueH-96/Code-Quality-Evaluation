def can_create_target_sheet(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X):
    # Create a large enough canvas to accommodate any placement of A and B
    max_dim = 50  # arbitrary large enough dimension to handle placements
    canvas = [['.' for _ in range(max_dim)] for _ in range(max_dim)]
    
    # Function to place a sheet onto the canvas
    def place_sheet(sheet, top_left_row, top_left_col):
        H = len(sheet)
        W = len(sheet[0])
        for i in range(H):
            for j in range(W):
                if sheet[i][j] == '#':
                    canvas[top_left_row + i][top_left_col + j] = '#'
    
    # Function to extract a subgrid from the canvas
    def extract_subgrid(top_left_row, top_left_col, H, W):
        return [canvas[top_left_row + i][top_left_col:top_left_col + W] for i in range(H)]
    
    # Try placing A and B in all possible positions on the canvas
    for a_row in range(max_dim - H_A + 1):
        for a_col in range(max_dim - W_A + 1):
            # Reset canvas to all transparent
            canvas = [['.' for _ in range(max_dim)] for _ in range(max_dim)]
            
            # Place A at (a_row, a_col)
            place_sheet(A, a_row, a_col)
            
            for b_row in range(max_dim - H_B + 1):
                for b_col in range(max_dim - W_B + 1):
                    # Create a copy of the canvas with only A placed
                    original_canvas = [row[:] for row in canvas]
                    
                    # Place B at (b_row, b_col)
                    place_sheet(B, b_row, b_col)
                    
                    # Check all possible top-left corners for extracting H_X x W_X
                    for x_row in range(max_dim - H_X + 1):
                        for x_col in range(max_dim - W_X + 1):
                            subgrid = extract_subgrid(x_row, x_col, H_X, W_X)
                            if subgrid == X:
                                return "Yes"
                    
                    # Restore the canvas to only having A placed
                    canvas = original_canvas
    
    return "No"

import sys
input = sys.stdin.read
data = input().split()

# Parse input
index = 0
H_A = int(data[index])
W_A = int(data[index + 1])
index += 2
A = [data[index + i] for i in range(H_A)]
index += H_A

H_B = int(data[index])
W_B = int(data[index + 1])
index += 2
B = [data[index + i] for i in range(H_B)]
index += H_B

H_X = int(data[index])
W_X = int(data[index + 1])
index += 2
X = [data[index + i] for i in range(H_X)]
index += H_X

# Solve the problem
result = can_create_target_sheet(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X)
print(result)