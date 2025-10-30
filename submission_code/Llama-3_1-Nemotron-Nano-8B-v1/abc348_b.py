n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y, i + 1))

for i in range(n):
    current_x, current_y, current_id = points[i]
    max_dist_sq = -1
    best_ids = []
    for j in range(n):
        if i == j:
            continue
        xj, yj, jid = points[j]
        dx = current_x - xj
        dy = current_y - yj
        dist_sq = dx ** 2 + dy ** 2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            best_ids = [jid]
        elif dist_sq == max_dist_sq:
            best_ids.append(jid)
    print(min(best_ids))