from collections import defaultdict

h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

present = [[True for _ in range(w)] for _ in range(h)]

freq_row = []
sum_row = []
len_row = []
for i in range(h):
    fr = defaultdict(int)
    for j in range(w):
        fr[grid[i][j]] += 1
    freq_row.append(fr)
    sum_row.append(sum(fr.values()))
    len_row.append(len(fr))

freq_col = []
sum_col = []
len_col = []
for j in range(w):
    fc = defaultdict(int)
    for i in range(h):
        fc[grid[i][j]] += 1
    freq_col.append(fc)
    sum_col.append(sum(fc.values()))
    len_col.append(len(fc))

rows_to_check = set(range(h))
cols_to_check = set(range(w))

while True:
    row_marked = []
    for i in rows_to_check:
        if sum_row[i] >= 2 and len_row[i] == 1:
            row_marked.append(i)
    col_marked = []
    for j in cols_to_check:
        if sum_col[j] >= 2 and len_col[j] == 1:
            col_marked.append(j)
    if not row_marked and not col_marked:
        break

    marked = set()
    for i in row_marked:
        for j in range(w):
            if present[i][j]:
                marked.add((i, j))
    for j in col_marked:
        for i in range(h):
            if present[i][j]:
                marked.add((i, j))

    if not marked:
        break

    modified_rows = set()
    modified_cols = set()
    for (i, j) in marked:
        present[i][j] = False
        color = grid[i][j]

        # Update row i
        freq_row[i][color] -= 1
        if freq_row[i][color] == 0:
            del freq_row[i][color]
            len_row[i] -= 1
        sum_row[i] -= 1

        # Update column j
        freq_col[j][color] -= 1
        if freq_col[j][color] == 0:
            del freq_col[j][color]
            len_col[j] -= 1
        sum_col[j] -= 1

        modified_rows.add(i)
        modified_cols.add(j)

    rows_to_check = modified_rows
    cols_to_check = modified_cols

count = 0
for i in range(h):
    for j in range(w):
        if present[i][j]:
            count += 1
print(count)