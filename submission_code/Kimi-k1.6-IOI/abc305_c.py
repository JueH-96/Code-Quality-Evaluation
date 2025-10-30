h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

cookies = []
for i in range(h):
    for j in range(w):
        if grid[i][j] == '#':
            cookies.append((i+1, j+1))  # Convert to 1-based indices

if not cookies:
    print(1, 1)  # Though problem states answer exists

a = min(i for i, j in cookies)
b = max(i for i, j in cookies)
c = min(j for i, j in cookies)
d = max(j for i, j in cookies)

missing = None
for i in range(a, b+1):
    for j in range(c, d+1):
        if grid[i-1][j-1] == '.':
            missing = (i, j)
            break
    if missing:
        break

print(missing[0], missing[1])