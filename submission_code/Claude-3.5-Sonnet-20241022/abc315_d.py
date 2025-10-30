def check_row(grid, H, W, marked):
    found = False
    for i in range(H):
        cookies = []
        for j in range(W):
            if grid[i][j] != '.':
                cookies.append((grid[i][j], i, j))
        if len(cookies) >= 2:
            color = cookies[0][0]
            all_same = all(c[0] == color for c in cookies)
            if all_same:
                found = True
                for _, r, c in cookies:
                    marked[r][c] = True
    return found

def check_col(grid, H, W, marked):
    found = False
    for j in range(W):
        cookies = []
        for i in range(H):
            if grid[i][j] != '.':
                cookies.append((grid[i][j], i, j))
        if len(cookies) >= 2:
            color = cookies[0][0]
            all_same = all(c[0] == color for c in cookies)
            if all_same:
                found = True
                for _, r, c in cookies:
                    marked[r][c] = True
    return found

def remove_marked(grid, marked, H, W):
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                grid[i][j] = '.'

def count_remaining(grid, H, W):
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '.':
                count += 1
    return count

H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input()))

while True:
    marked = [[False] * W for _ in range(H)]
    row_found = check_row(grid, H, W, marked)
    col_found = check_col(grid, H, W, marked)
    
    if not row_found and not col_found:
        break
        
    remove_marked(grid, marked, H, W)

print(count_remaining(grid, H, W))