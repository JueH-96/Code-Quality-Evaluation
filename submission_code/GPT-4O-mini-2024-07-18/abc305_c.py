def find_eaten_cookie(H, W, grid):
    # Initialize the boundaries of the rectangle
    top = H
    bottom = -1
    left = W
    right = -1
    
    # Scan the grid to find the boundaries of the rectangle with cookies
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < top:
                    top = i
                if i > bottom:
                    bottom = i
                if j < left:
                    left = j
                if j > right:
                    right = j
    
    # Now we have the rectangle defined by (top, left) to (bottom, right)
    # We need to find the missing cookie in this rectangle
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1  # Convert to 1-based index

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
grid = data[1:H + 1]

# Find and print the position of the eaten cookie
result = find_eaten_cookie(H, W, grid)
print(result[0], result[1])