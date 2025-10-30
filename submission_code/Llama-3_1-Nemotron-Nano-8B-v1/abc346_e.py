import bisect
from collections import defaultdict

H, W, M = map(int, input().split())

t_row = [0] * (H + 1)
c_row = [0] * (H + 1)
t_col = [0] * (W + 1)
c_col = [0] * (W + 1)

for step in range(1, M + 1):
    T, A, X = map(int, input().split())
    if T == 1:
        t_row[A] = step
        c_row[A] = X
    else:
        t_col[A] = step
        c_col[A] = X

sorted_t_col = sorted(t_col[j] for j in range(1, W + 1))
sorted_t_row = sorted(t_row[i] for i in range(1, H + 1))

cnt_rows_zero = sum(1 for i in range(1, H + 1) if t_row[i] == 0)
cnt_cols_zero = sum(1 for j in range(1, W + 1) if t_col[j] == 0)
case1 = cnt_rows_zero * cnt_cols_zero

case2 = 0
for i in range(1, H + 1):
    if c_row[i] == 0:
        t = t_row[i]
        case2 += bisect.bisect_right(sorted_t_col, t)

case3 = 0
for j in range(1, W + 1):
    if c_col[j] == 0:
        s = t_col[j]
        case3 += bisect.bisect_right(sorted_t_row, s)

color0_count = case2 + case3 - case1

color_counts = defaultdict(int)

for i in range(1, H + 1):
    x = c_row[i]
    if x != 0:
        t = t_row[i]
        cnt = bisect.bisect_right(sorted_t_col, t)
        color_counts[x] += cnt

for j in range(1, W + 1):
    y = c_col[j]
    if y != 0:
        s = t_col[j]
        cnt = bisect.bisect_right(sorted_t_row, s)
        color_counts[y] += cnt

if color0_count > 0:
    color_counts[0] = color0_count

sorted_colors = sorted(color_counts.keys())
K = len(sorted_colors)
print(K)
for c in sorted_colors:
    print(c, color_counts[c])