import bisect

n, m, Sx, Sy = map(int, input().split())
houses = [tuple(map(int, input().split())) for _ in range(n)]
moves = [input().split() for _ in range(m)]

current_x, current_y = Sx, Sy
vertical = {}
horizontal = {}
line_segments = []

for d, c in moves:
    c = int(c)
    new_x, new_y = current_x, current_y
    if d == 'U':
        new_y = current_y + c
        line_segments.append(('vertical', current_x, min(current_y, new_y), max(current_y, new_y)))
    elif d == 'D':
        new_y = current_y - c
        line_segments.append(('vertical', current_x, min(current_y, new_y), max(current_y, new_y)))
    elif d == 'L':
        new_x = current_x - c
        line_segments.append(('horizontal', current_y, min(current_x, new_x), max(current_x, new_x)))
    elif d == 'R':
        new_x = current_x + c
        line_segments.append(('horizontal', current_y, min(current_x, new_x), max(current_x, new_x)))
    current_x, current_y = new_x, new_y

vertical_intervals = {}
for seg in line_segments:
    if seg[0] == 'vertical':
        x, y1, y2 = seg[1], seg[2], seg[3]
        if x not in vertical_intervals:
            vertical_intervals[x] = []
        vertical_intervals[x].append((y1, y2))

for x in vertical_intervals:
    intervals = sorted(vertical_intervals[x])
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    starts = [interval[0] for interval in merged]
    vertical_intervals[x] = (merged, starts)

horizontal_intervals = {}
for seg in line_segments:
    if seg[0] == 'horizontal':
        y, x1, x2 = seg[1], seg[2], seg[3]
        if y not in horizontal_intervals:
            horizontal_intervals[y] = []
        horizontal_intervals[y].append((x1, x2))

for y in horizontal_intervals:
    intervals = sorted(horizontal_intervals[y])
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    starts = [interval[0] for interval in merged]
    horizontal_intervals[y] = (merged, starts)

count = 0
for x, y in houses:
    found = False
    if x in vertical_intervals:
        merged, starts = vertical_intervals[x]
        idx = bisect.bisect_right(starts, y) - 1
        if idx >= 0 and merged[idx][1] >= y:
            found = True
    if not found and y in horizontal_intervals:
        merged, starts = horizontal_intervals[y]
        idx = bisect.bisect_right(starts, x) - 1
        if idx >= 0 and merged[idx][1] >= x:
            found = True
    if found:
        count += 1

print(current_x, current_y, count)