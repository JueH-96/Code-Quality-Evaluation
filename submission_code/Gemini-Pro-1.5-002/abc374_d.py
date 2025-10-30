import itertools
import math

def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

n, s, t = map(int, input().split())
segments = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    segments.append(((a, b), (c, d)))

min_time = float('inf')

for perm in itertools.permutations(range(n)):
    for starts in itertools.product([0, 1], repeat=n):
        current_time = 0
        current_x, current_y = 0, 0
        for i in range(n):
            idx = perm[i]
            start_point = segments[idx][starts[i]]
            end_point = segments[idx][1 - starts[i]]
            current_time += dist(current_x, current_y, start_point[0], start_point[1]) / s
            current_x, current_y = start_point[0], start_point[1]
            current_time += dist(current_x, current_y, end_point[0], end_point[1]) / t
            current_x, current_y = end_point[0], end_point[1]
        min_time = min(min_time, current_time)

print(min_time)