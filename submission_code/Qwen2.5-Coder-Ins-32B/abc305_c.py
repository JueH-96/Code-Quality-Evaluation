# YOUR CODE HERE
import sys

def find_eaten_cookie(H, W, grid):
    # Find the top-left corner (a, c) of the rectangle
    a, c = -1, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                a, c = i, j
                break
        if a != -1:
            break
    
    # Find the bottom-right corner (b, d) of the rectangle
    b, d = -1, -1
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if grid[i][j] == '#':
                b, d = i, j
                break
        if b != -1:
            break
    
    # Find the empty square (i, j) inside the rectangle
    for i in range(a, b + 1):
        for j in range(c, d + 1):
            if grid[i][j] == '.':
                return i + 1, j + 1

# Read input
input = sys.stdin.read().split()
H, W = int(input[0]), int(input[1])
grid = input[2:]

# Find and print the eaten cookie position
result = find_eaten_cookie(H, W, grid)
print(result[0], result[1])