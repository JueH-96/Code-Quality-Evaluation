n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

for i in range(n):
    max_dist_sq = -1
    candidates = []
    x1, y1 = points[i]
    for j in range(n):
        if i == j:
            continue
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            candidates = [j]
        elif dist_sq == max_dist_sq:
            candidates.append(j)
    min_j = min(candidates)
    print(min_j + 1)