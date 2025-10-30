def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    
    while True:
        marked = [[False] * W for _ in range(H)]
        any_marked = False
        
        # Check rows
        for i in range(H):
            remaining_cookies = []
            remaining_positions = []
            for j in range(W):
                if grid[i][j] != '.':
                    remaining_cookies.append(grid[i][j])
                    remaining_positions.append(j)
            
            if len(remaining_cookies) >= 2 and len(set(remaining_cookies)) == 1:
                # All remaining cookies in this row have the same color
                for j in remaining_positions:
                    marked[i][j] = True
                    any_marked = True
        
        # Check columns
        for j in range(W):
            remaining_cookies = []
            remaining_positions = []
            for i in range(H):
                if grid[i][j] != '.':
                    remaining_cookies.append(grid[i][j])
                    remaining_positions.append(i)
            
            if len(remaining_cookies) >= 2 and len(set(remaining_cookies)) == 1:
                # All remaining cookies in this column have the same color
                for i in remaining_positions:
                    marked[i][j] = True
                    any_marked = True
        
        if not any_marked:
            break
        
        # Remove marked cookies
        for i in range(H):
            for j in range(W):
                if marked[i][j]:
                    grid[i][j] = '.'
    
    # Count remaining cookies
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '.':
                count += 1
    
    print(count)

solve()