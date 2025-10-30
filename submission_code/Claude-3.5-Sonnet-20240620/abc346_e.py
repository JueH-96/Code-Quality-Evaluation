# YOUR CODE HERE
H, W, M = map(int, input().split())

rows = [0] * H
cols = [0] * W
row_last_op = [-1] * H
col_last_op = [-1] * W

operations = []
for i in range(M):
    T, A, X = map(int, input().split())
    operations.append((T, A-1, X))  # Convert to 0-based indexing

colors = {}
last_op = -1

for i, (T, A, X) in enumerate(operations):
    if T == 1:
        rows[A] = X
        row_last_op[A] = i
    else:
        cols[A] = X
        col_last_op[A] = i
    last_op = max(last_op, i)

for i in range(H):
    if row_last_op[i] == last_op:
        color = rows[i]
        colors[color] = colors.get(color, 0) + W
    else:
        for j in range(W):
            if col_last_op[j] > row_last_op[i]:
                color = cols[j]
            else:
                color = rows[i]
            colors[color] = colors.get(color, 0) + 1

print(len(colors))
for color in sorted(colors.keys()):
    print(f"{color} {colors[color]}")