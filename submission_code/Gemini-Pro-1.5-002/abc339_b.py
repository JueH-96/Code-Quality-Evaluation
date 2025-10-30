# YOUR CODE HERE
def solve():
    h, w, n = map(int, input().split())
    grid = [['.' for _ in range(w)] for _ in range(h)]
    
    r, c = 0, 0
    dr, dc = -1, 0
    
    for _ in range(n):
        if grid[r][c] == '.':
            grid[r][c] = '#'
            dr, dc = dc, -dr
        else:
            grid[r][c] = '.'
            dr, dc = -dc, dr
            
        r = (r + dr) % h
        c = (c + dc) % w
        
    for row in grid:
        print("".join(row))

solve()