# YOUR CODE HERE
def remove_cookies(grid):
    H, W = len(grid), len(grid[0])
    marked = [[False] * W for _ in range(H)]
    
    def mark_row(i):
        if len(set(grid[i])) == 1 and grid[i][0] != '.':
            for j in range(W):
                marked[i][j] = True
    
    def mark_col(j):
        col = [grid[i][j] for i in range(H)]
        if len(set(col)) == 1 and col[0] != '.':
            for i in range(H):
                marked[i][j] = True
    
    changed = True
    while changed:
        changed = False
        for i in range(H):
            mark_row(i)
        for j in range(W):
            mark_col(j)
        
        for i in range(H):
            for j in range(W):
                if marked[i][j]:
                    grid[i][j] = '.'
                    marked[i][j] = False
                    changed = True
    
    return sum(row.count(c) for row in grid for c in row if c != '.')

H, W = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]

print(remove_cookies(grid))