def count_remaining_cookies():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    
    # Convert grid to a list of lists for easier manipulation
    grid = [list(row) for row in grid]
    
    while True:
        marked = set()
        
        # Check rows
        for i in range(H):
            if len(set(grid[i])) == 1:  # All elements in the row are the same
                for j in range(W):
                    if grid[i][j] != '.':
                        marked.add((i, j))
        
        # Check columns
        for j in range(W):
            column = [grid[i][j] for i in range(H) if grid[i][j] != '.']
            if len(column) > 1 and len(set(column)) == 1:  # All elements in the column are the same
                for i in range(H):
                    if grid[i][j] != '.':
                        marked.add((i, j))
        
        # If no cookies are marked, break the loop
        if not marked:
            break
        
        # Remove marked cookies
        for i, j in marked:
            grid[i][j] = '.'
    
    # Count remaining cookies
    remaining_cookies = sum(row.count('.') == 0 for row in grid for _ in row)
    
    print(remaining_cookies)

# To run the function, uncomment the following line:
# count_remaining_cookies()