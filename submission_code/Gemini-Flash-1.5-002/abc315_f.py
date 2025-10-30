import itertools
import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

min_dist = float('inf')

for i in range(1 << (N - 2)):
    path = [points[0]]
    skipped = 0
    for j in range(N - 2):
        if (i >> j) & 1:
            path.append(points[j + 1])
        else:
            skipped += 1
    path.append(points[-1])
    
    total_dist = 0
    for k in range(len(path) - 1):
        total_dist += dist(path[k], path[k+1])
    
    penalty = 0
    if skipped > 0:
        penalty = 2**(skipped)
    
    min_dist = min(min_dist, total_dist + penalty)

print(min_dist)