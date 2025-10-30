H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find the top boundary (a)
a = 0
while a < H and '#' not in grid[a]:
    a += 1

# Find the bottom boundary (b)
b = H - 1
while b >= 0 and '#' not in grid[b]:
    b -= 1

# Find the left boundary (c)
c = 0
while c < W:
    found = False
    for i in range(H):
        if grid[i][c] == '#':
            found = True
            break
    if found:
        break
    c += 1

# Find the right boundary (d)
d = W - 1
while d >= 0:
    found = False
    for i in range(H):
        if grid[i][d] == '#':
            found = True
            break
    if found:
        break
    d -= 1

# Search for the missing cookie
for i in range(a, b + 1):
    for j in range(c, d + 1):
        if grid[i][j] == '.':
            print(i + 1, j + 1)
            exit()