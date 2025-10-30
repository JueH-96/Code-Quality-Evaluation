import sys
import bisect
from collections import defaultdict

H, W, M = map(int, sys.stdin.readline().split())

row_time = [0] * H
row_color = [0] * H
column_time = [0] * W
column_color = [0] * W

for i in range(1, M + 1):
    t, a, x = map(int, sys.stdin.readline().split())
    if t == 1:
        idx = a - 1
        row_time[idx] = i
        row_color[idx] = x
    else:
        idx = a - 1
        column_time[idx] = i
        column_color[idx] = x

sorted_col_times = sorted(column_time)
sorted_row_times = sorted(row_time)

rows_unpainted = sum(1 for t in row_time if t == 0)
columns_unpainted = sum(1 for t in column_time if t == 0)
count_zero = rows_unpainted * columns_unpainted

freq = defaultdict(int)
if count_zero > 0:
    freq[0] += count_zero

for i in range(H):
    rt = row_time[i]
    if rt == 0:
        continue
    color = row_color[i]
    cnt = bisect.bisect_left(sorted_col_times, rt)
    freq[color] += cnt

for j in range(W):
    ct = column_time[j]
    if ct == 0:
        continue
    color = column_color[j]
    cnt = bisect.bisect_left(sorted_row_times, ct)
    freq[color] += cnt

colors = sorted(freq.keys())
print(len(colors))
for c in colors:
    print(c, freq[c])