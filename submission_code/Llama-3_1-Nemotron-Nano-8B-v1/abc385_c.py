import bisect
from collections import defaultdict

n, *rest = map(int, open(0).read().split())
H = rest[:n]

height_positions = defaultdict(list)
for idx, h in enumerate(H, 1):
    height_positions[h].append(idx)

max_count = 0

for h, positions in height_positions.items():
    m = len(positions)
    if m == 0:
        continue
    positions.sort()
    min_p = positions[0]
    max_p = positions[-1]
    max_d = max_p - min_p
    current_max = 1

    for d in range(1, max_d + 1):
        groups = defaultdict(list)
        for pos in positions:
            rem = pos % d
            groups[rem].append(pos)
        group_max = 0
        for group in groups.values():
            if not group:
                continue
            run = 1
            max_run = 1
            for i in range(1, len(group)):
                if group[i] - group[i-1] == d:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 1
            if max_run > group_max:
                group_max = max_run
        if group_max > current_max:
            current_max = group_max
    if current_max > max_count:
        max_count = current_max

print(max_count)