H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

# Track which cookies are present
present = [[True] * W for _ in range(H)]

while True:
    marked = [[False] * W for _ in range(H)]
    
    # Check rows
    for i in range(H):
        row_cookies = []
        for j in range(W):
            if present[i][j]:
                row_cookies.append(grid[i][j])
        
        if len(row_cookies) >= 2 and all(c == row_cookies[0] for c in row_cookies):
            for j in range(W):
                if present[i][j]:
                    marked[i][j] = True
    
    # Check columns
    for j in range(W):
        col_cookies = []
        for i in range(H):
            if present[i][j]:
                col_cookies.append(grid[i][j])
        
        if len(col_cookies) >= 2 and all(c == col_cookies[0] for c in col_cookies):
            for i in range(H):
                if present[i][j]:
                    marked[i][j] = True
    
    # Remove marked cookies
    any_marked = False
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                present[i][j] = False
                any_marked = True
    
    if not any_marked:
        break

# Count remaining
count = sum(sum(row) for row in present)
print(count)