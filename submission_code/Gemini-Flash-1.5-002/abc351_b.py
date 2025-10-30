# YOUR CODE HERE
n = int(input())
grid_a = []
for _ in range(n):
    grid_a.append(input())
grid_b = []
for _ in range(n):
    grid_b.append(input())

diff_row = -1
diff_col = -1

for i in range(n):
    for j in range(n):
        if grid_a[i][j] != grid_b[i][j]:
            diff_row = i + 1
            diff_col = j + 1
            break
    if diff_row != -1:
        break

print(diff_row, diff_col)