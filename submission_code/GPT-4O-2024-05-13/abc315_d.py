# YOUR CODE HERE
def remove_cookies(H, W, grid):
    while True:
        marked = set()
        
        # Check rows
        for i in range(H):
            row = [grid[i][j] for j in range(W) if (i, j) not in marked]
            if len(row) > 1 and all(c == row[0] for c in row):
                for j in range(W):
                    if (i, j) not in marked:
                        marked.add((i, j))
        
        # Check columns
        for j in range(W):
            col = [grid[i][j] for i in range(H) if (i, j) not in marked]
            if len(col) > 1 and all(c == col[0] for c in col):
                for i in range(H):
                    if (i, j) not in marked:
                        marked.add((i, j))
        
        if not marked:
            break
        
        for i, j in marked:
            grid[i][j] = '.'
    
    remaining_cookies = sum(1 for i in range(H) for j in range(W) if grid[i][j] != '.')
    return remaining_cookies

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = [list(data[i + 2]) for i in range(H)]

print(remove_cookies(H, W, grid))