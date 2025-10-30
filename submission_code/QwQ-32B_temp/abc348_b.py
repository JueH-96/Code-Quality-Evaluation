n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(n):
    max_dist = -1
    best_id = -1
    for j in range(n):
        if i == j:
            continue
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist:
            max_dist = dist_sq
            best_id = j + 1
        elif dist_sq == max_dist:
            if (j + 1) < best_id:
                best_id = j + 1
    print(best_id)