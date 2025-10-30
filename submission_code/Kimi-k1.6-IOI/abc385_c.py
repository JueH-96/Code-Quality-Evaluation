import bisect
from collections import defaultdict

n = int(input())
H = list(map(int, input().split()))

pos_dict = defaultdict(list)
for idx, h in enumerate(H):
    pos_dict[h].append(idx)

max_answer = 1

for heights in pos_dict.values():
    m = len(heights)
    if m == 1:
        current_max = 1
    else:
        current_max = 1
        for i in range(m):
            for j in range(i + 1, m):
                d = heights[j] - heights[i]
                current_len = 2
                next_val = heights[j] + d
                k = bisect.bisect_left(heights, next_val, j + 1)
                while k < m and heights[k] == next_val:
                    current_len += 1
                    next_val += d
                    k = bisect.bisect_left(heights, next_val, k + 1)
                if current_len > current_max:
                    current_max = current_len
    if current_max > max_answer:
        max_answer = current_max

print(max_answer)