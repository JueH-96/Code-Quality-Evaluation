def solve():
    H, W = map(int, input().split())
    cookies = []
    for _ in range(H):
        cookies.append(list(input().strip()))
    
    while True:
        marked = [[False for _ in range(W)] for _ in range(H)]
        any_marked = False
        
        # Mark in rows
        for i in range(H):
            cookies_in_row = [(j, cookies[i][j]) for j in range(W) if cookies[i][j] != '.']
            if len(cookies_in_row) >= 2:
                colors = set(color for _, color in cookies_in_row)
                if len(colors) == 1:  # All cookies in the row have the same color
                    for j, _ in cookies_in_row:
                        marked[i][j] = True
                    any_marked = True
        
        # Mark in columns
        for j in range(W):
            cookies_in_col = [(i, cookies[i][j]) for i in range(H) if cookies[i][j] != '.']
            if len(cookies_in_col) >= 2:
                colors = set(color for _, color in cookies_in_col)
                if len(colors) == 1:  # All cookies in the column have the same color
                    for i, _ in cookies_in_col:
                        marked[i][j] = True
                    any_marked = True
        
        # Remove marked cookies
        if any_marked:
            for i in range(H):
                for j in range(W):
                    if marked[i][j]:
                        cookies[i][j] = '.'
        else:
            break
    
    # Count remaining cookies
    remaining = sum(1 for i in range(H) for j in range(W) if cookies[i][j] != '.')
    return remaining

print(solve())