n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]

for i in range(n):
    max_dist_sq = -1
    best_id = 0
    x1, y1 = points[i]
    for j in range(n):
        if j == i:
            continue
        x2, y2 = points[j]
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx ** 2 + dy ** 2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            best_id = j + 1
        elif dist_sq == max_dist_sq:
            if (j + 1) < best_id:
                best_id = j + 1
    print(best_id)