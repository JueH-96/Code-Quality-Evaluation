def solve():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    
    while True:
        marked = set()
        
        # Mark rows
        for r in range(H):
            row = grid[r]
            if len(row) >= 2 and all(row[0] == cell for cell in row):
                for c in range(len(row)):
                    marked.add((r, c))
        
        # Mark columns
        for c in range(W):
            col = []
            for r in range(H):
                if c < len(grid[r]):
                    col.append(grid[r][c])
            
            if len(col) >= 2 and all(col[0] == cell for cell in col):
                for r in range(H):
                    if c < len(grid[r]):
                        marked.add((r, c))
        
        if not marked:
            break
        
        # Remove marked cookies
        new_grid = []
        for r in range(H):
            new_row = []
            for c in range(len(grid[r])):
                if (r, c) not in marked:
                    new_row.append(grid[r][c])
            new_grid.append(new_row)
        grid = new_grid
    
    count = 0
    for row in grid:
        count += len(row)
    print(count)

solve()