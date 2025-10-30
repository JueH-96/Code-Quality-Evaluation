def can_form_rectangle(H, W, grid):
    # Find the bounds of the black cells
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    # Check if all cells in the bounding rectangle can be painted black
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                return "No"
    
    # Check if there are any '?' in the bounding rectangle
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '?':
                continue  # '?' can be painted either way
            if grid[i][j] == '#':
                continue  # already black

    return "Yes"

import sys

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = data[1:H + 1]
    
    result = can_form_rectangle(H, W, grid)
    print(result)

if __name__ == "__main__":
    main()