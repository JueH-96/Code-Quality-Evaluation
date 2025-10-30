# YOUR CODE HERE
def find_missing_cookie(H, W, grid):
    top, bottom, left, right = None, None, None, None
    
    # Find the boundaries of the rectangle
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if top is None:
                    top = i
                bottom = i
                if left is None:
                    left = j
                right = j
    
    # Check for the missing cookie
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':  # This is the missing cookie
                return i + 1, j + 1  # +1 to convert from 0-indexed to 1-indexed

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find and print the missing cookie
i, j = find_missing_cookie(H, W, grid)
print(i, j)