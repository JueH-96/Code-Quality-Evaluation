n = int(input())

matrix = [[0]*n for _ in range(n)]
dr = [0, 1, 0, -1]
dc = [1, 0, -1, 0]
r, c, dir = 0, 0, 0

for num in range(1, n*n + 1):
    matrix[r][c] = num
    nr = r + dr[dir]
    nc = c + dc[dir]
    if 0 <= nr < n and 0 <= nc < n and matrix[nr][nc] == 0:
        r, c = nr, nc
    else:
        dir = (dir + 1) % 4
        r = r + dr[dir]
        c = c + dc[dir]

center = ((n-1)//2, (n-1)//2)
for i in range(n):
    for j in range(n):
        if i == center[0] and j == center[1]:
            matrix[i][j] = 'T'
        else:
            matrix[i][j] = str(matrix[i][j])

for row in matrix:
    print(' '.join(row))