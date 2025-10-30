# YOUR CODE HERE
def find_eaten_cookie(H, W, grid):
    # Find the top-left and bottom-right corners of the rectangle
    top, left, bottom, right = H, W, 0, 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                top = min(top, i)
                left = min(left, j)
                bottom = max(bottom, i)
                right = max(right, j)
    
    # Now we know the rectangle is from (top, left) to (bottom, right)
    # Find the missing cookie
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1

# Read input
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = [data[i + 2] for i in range(H)]

# Find the eaten cookie
result = find_eaten_cookie(H, W, grid)
print(result[0], result[1])