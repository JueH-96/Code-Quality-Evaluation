H, W, D = map(int, input().split())
floor_cells = []
for i in range(H):
    s = input().strip()
    for j in range(W):
        if s[j] == '.':
            floor_cells.append((i + 1, j + 1))

max_count = 0
n = len(floor_cells)
for i in range(n):
    for j in range(i + 1, n):
        a = floor_cells[i]
        b = floor_cells[j]
        count = 0
        for (x, y) in floor_cells:
            d1 = abs(x - a[0]) + abs(y - a[1])
            d2 = abs(x - b[0]) + abs(y - b[1])
            if d1 <= D or d2 <= D:
                count += 1
        if count > max_count:
            max_count = count

print(max_count)