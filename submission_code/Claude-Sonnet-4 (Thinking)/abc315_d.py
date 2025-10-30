H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

while True:
    marked = [[False] * W for _ in range(H)]
    
    # Check rows
    for i in range(H):
        row_cookies = []
        for j in range(W):
            if grid[i][j] != '.':
                row_cookies.append(grid[i][j])
        
        if len(row_cookies) >= 2 and all(color == row_cookies[0] for color in row_cookies):
            for j in range(W):
                if grid[i][j] != '.':
                    marked[i][j] = True
    
    # Check columns
    for j in range(W):
        col_cookies = []
        for i in range(H):
            if grid[i][j] != '.':
                col_cookies.append(grid[i][j])
        
        if len(col_cookies) >= 2 and all(color == col_cookies[0] for color in col_cookies):
            for i in range(H):
                if grid[i][j] != '.':
                    marked[i][j] = True
    
    # Remove marked cookies
    any_marked = False
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                grid[i][j] = '.'
                any_marked = True
    
    if not any_marked:
        break

# Count remaining cookies
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] != '.':
            count += 1

print(count)