# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    
    def check_row(r):
        if len(grid[r]) < 2:
            return []
        
        first = next((c for c in grid[r] if c != '.'), None)
        if all(c == first or c == '.' for c in grid[r]):
            return [(r, c) for c in range(w) if grid[r][c] != '.']
        else:
            return []

    def check_col(c):
        col_vals = [grid[r][c] for r in range(h) if grid[r][c] != '.']
        if len(col_vals) < 2:
            return []
        
        first = col_vals[0]
        if all(v == first for v in col_vals):
            return [(r, c) for r in range(h) if grid[r][c] != '.']
        else:
            return []

    while True:
        marked = set()
        for r in range(h):
            marked.update(check_row(r))
        for c in range(w):
            marked.update(check_col(c))

        if not marked:
            break

        for r, c in marked:
            grid[r][c] = '.'
            
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != '.':
                count += 1
    print(count)

solve()