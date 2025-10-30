# YOUR CODE HERE
def can_form_rectangle(H, W, grid):
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    
    # Find the bounding box of black cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    
    # Check the validity of the rectangle
    for i in range(H):
        for j in range(W):
            if min_row <= i <= max_row and min_col <= j <= max_col:
                # Inside the rectangle bounds
                if grid[i][j] == '.':
                    return "No"
            else:
                # Outside the rectangle bounds
                if grid[i][j] == '#':
                    return "No"
    
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:H+2]

print(can_form_rectangle(H, W, grid))