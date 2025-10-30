import sys

def find_missing_cookie(h, w, grid):
    top, bottom, left, right = None, None, None, None
    
    # Find the boundaries of the rectangle with cookies
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                if top is None:
                    top = i
                bottom = i
                if left is None:
                    left = j
                right = j
    
    # Find the missing cookie within the rectangle
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1

# Read input
h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

# Find and print the missing cookie
row, col = find_missing_cookie(h, w, grid)
print(row, col)