# YOUR CODE HERE
n = int(input())
rectangles = [tuple(map(int, input().split())) for _ in range(n)]
x_intervals = [(a, b) for a, b, _, _ in rectangles]
y_intervals = [(c, d) for _, _, c, d in rectangles]
x_intervals.sort()
y_intervals.sort()
x_intervals = [(x, y) for x, y in x_intervals if x < y]
y_intervals = [(x, y) for x, y in y_intervals if x < y]
x_start, x_end = x_intervals[0]
y_start, y_end = y_intervals[0]
area = 0
for x_start, x_end in x_intervals:
    for y_start, y_end in y_intervals:
        if x_start >= x_end or y_start >= y_end:
            continue
        x_overlap = min(x_end, y_end) - max(x_start, y_start)
        if x_overlap > 0:
            area += x_overlap * (y_end - y_start)
print(area)