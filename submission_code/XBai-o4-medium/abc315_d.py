H, W = map(int, input().split())
c = [input().strip() for _ in range(H)]
row_removed = [False] * H
col_removed = [False] * W

while True:
    to_remove_rows = []
    for i in range(H):
        if row_removed[i]:
            continue
        chars = [c[i][j] for j in range(W) if not col_removed[j]]
        if len(chars) >= 2 and len(set(chars)) == 1:
            to_remove_rows.append(i)
    to_remove_cols = []
    for j in range(W):
        if col_removed[j]:
            continue
        chars = [c[i][j] for i in range(H) if not row_removed[i]]
        if len(chars) >= 2 and len(set(chars)) == 1:
            to_remove_cols.append(j)
    if not to_remove_rows and not to_remove_cols:
        break
    for i in to_remove_rows:
        row_removed[i] = True
    for j in to_remove_cols:
        col_removed[j] = True

res = 0
for i in range(H):
    if row_removed[i]:
        continue
    for j in range(W):
        if not col_removed[j]:
            res += 1
print(res)