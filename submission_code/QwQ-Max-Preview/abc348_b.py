n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

for i in range(n):
    max_dist_sq = -1
    farthest = []
    for j in range(n):
        if i == j:
            continue
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            farthest = [j + 1]
        elif dist_sq == max_dist_sq:
            farthest.append(j + 1)
    print(min(farthest))