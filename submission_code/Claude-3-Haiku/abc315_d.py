# Read input
H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

def remove_cookies():
    # Mark cookies in rows
    for i in range(H):
        row = grid[i]
        if len(set(row)) == 1 and len(row) > 1:
            for j in range(W):
                grid[i][j] = '.'
    
    # Mark cookies in columns
    for j in range(W):
        col = [grid[i][j] for i in range(H)]
        if len(set(col)) == 1 and len(col) > 1:
            for i in range(H):
                grid[i][j] = '.'

    # Check if any cookies were removed
    return any('.' in row for row in grid)

while remove_cookies():
    pass

# Count remaining cookies
print(sum(row.count('.') < W for row in grid))