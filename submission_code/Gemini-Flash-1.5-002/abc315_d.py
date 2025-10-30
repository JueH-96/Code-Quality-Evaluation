# YOUR CODE HERE
H, W = map(int, input().split())
cookies = [list(input()) for _ in range(H)]

while True:
    marked = [[False] * W for _ in range(H)]
    
    # Check rows
    for i in range(H):
        row = [c for c in cookies[i] if c != '.']
        if len(row) >= 2 and all(c == row[0] for c in row):
            for j in range(W):
                if cookies[i][j] != '.':
                    marked[i][j] = True

    # Check columns
    for j in range(W):
        col = [cookies[i][j] for i in range(H) if cookies[i][j] != '.']
        if len(col) >= 2 and all(c == col[0] for c in col):
            for i in range(H):
                if cookies[i][j] != '.':
                    marked[i][j] = True

    # Remove marked cookies
    removed = False
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                cookies[i][j] = '.'
                removed = True

    if not removed:
        break

count = 0
for i in range(H):
    for j in range(W):
        if cookies[i][j] != '.':
            count += 1

print(count)