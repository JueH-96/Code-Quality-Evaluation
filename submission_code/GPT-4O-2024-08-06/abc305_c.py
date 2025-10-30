# YOUR CODE HERE
def find_missing_cookie():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = data[1:H+1]
    
    # Initialize the bounds of the rectangle
    min_row = H
    max_row = 0
    min_col = W
    max_col = 0
    
    # Find the bounds of the rectangle containing all cookies
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
    
    # Now we have the rectangle from (min_row, min_col) to (max_row, max_col)
    # We need to find the single '.' in this rectangle
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                # Output the 1-based index of the missing cookie
                print(i + 1, j + 1)
                return