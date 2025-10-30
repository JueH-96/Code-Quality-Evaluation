H, W = map(int, input().split())
cookies = [list(input()) for _ in range(H)]
marked = [[False]*W for _ in range(H)]

while True:
    # Mark rows
    for i in range(H):
        if len(set(cookies[i])) == 1 and cookies[i][0] != '.':
            for j in range(W):
                marked[i][j] = True

    # Mark columns
    for j in range(W):
        column = [cookies[i][j] for i in range(H)]
        if len(set(column)) == 1 and column[0] != '.':
            for i in range(H):
                marked[i][j] = True

    # Check if any cookies are marked
    if any(any(row) for row in marked):
        # Remove marked cookies
        for i in range(H):
            for j in range(W):
                if marked[i][j]:
                    cookies[i][j] = '.'
                    marked[i][j] = False
    else:
        break

# Count remaining cookies
print(sum(row.count('.') for row in cookies))