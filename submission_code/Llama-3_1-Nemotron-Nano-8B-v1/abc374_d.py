import itertools

n, S, T = map(int, input().split())
segments = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    dx = c - a
    dy = d - b
    length = (dx**2 + dy**2)**0.5
    segments.append(((a, b), (c, d), length))

min_time = float('inf')

for perm in itertools.permutations(range(n)):
    for dirs in itertools.product([0, 1], repeat=n):
        current = (0.0, 0.0)
        total = 0.0
        for i in range(n):
            seg_idx = perm[i]
            seg = segments[seg_idx]
            direction = dirs[i]
            if direction == 0:
                start = seg[0]
                end = seg[1]
            else:
                start = seg[1]
                end = seg[0]
            dx = start[0] - current[0]
            dy = start[1] - current[1]
            dist = (dx**2 + dy**2)**0.5
            total += dist / S
            total += seg[2] / T
            current = end
        if total < min_time:
            min_time = total

print("{0:.20f}".format(min_time))