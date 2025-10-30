n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

for i in range(n):
    max_dist_sq = -1
    candidates = []
    for k in range(n):
        if k == i:
            continue
        dx = points[i][0] - points[k][0]
        dy = points[i][1] - points[k][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            candidates = [k + 1]
        elif dist_sq == max_dist_sq:
            candidates.append(k + 1)
    print(min(candidates))