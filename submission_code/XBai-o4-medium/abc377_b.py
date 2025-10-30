rows = [input().strip() for _ in range(8)]

row_has = [False] * 8
for i in range(8):
    row_has[i] = '#' in rows[i]

col_has = [False] * 8
for j in range(8):
    for i in range(8):
        if rows[i][j] == '#':
            col_has[j] = True
            break

count = 0
for i in range(8):
    for j in range(8):
        if rows[i][j] == '.' and not row_has[i] and not col_has[j]:
            count += 1

print(count)