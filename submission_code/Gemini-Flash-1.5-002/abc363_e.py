# YOUR CODE HERE
h, w, y = map(int, input().split())
a = []
for i in range(h):
    a.append(list(map(int, input().split())))

for year in range(1, y + 1):
    sea_level = year
    sunk = [[False for _ in range(w)] for _ in range(h)]
    q = []
    for i in range(h):
        if a[i][0] <= sea_level:
            q.append((i, 0))
            sunk[i][0] = True
        if a[i][w - 1] <= sea_level:
            q.append((i, w - 1))
            sunk[i][w - 1] = True
    for j in range(w):
        if a[0][j] <= sea_level:
            q.append((0, j))
            sunk[0][j] = True
        if a[h - 1][j] <= sea_level:
            q.append((h - 1, j))
            sunk[h - 1][j] = True
    while q:
        i, j = q.pop(0)
        for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
            if 0 <= x < h and 0 <= y < w and not sunk[x][y] and a[x][y] <= sea_level:
                sunk[x][y] = True
                q.append((x, y))
    count = 0
    for i in range(h):
        for j in range(w):
            if not sunk[i][j]:
                count += 1
    print(count)