n = int(input())
points = []
for i in range(n):
    x, y = map(int, input().split())
    points.append((x, y, i + 1))

for i in range(n):
    current_x, current_y, current_id = points[i]
    max_dist_sq = -1
    candidates = []
    for j in range(n):
        if j == i:
            continue
        j_x, j_y, j_id = points[j]
        dx = current_x - j_x
        dy = current_y - j_y
        dist_sq = dx ** 2 + dy ** 2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            candidates = [j_id]
        elif dist_sq == max_dist_sq:
            candidates.append(j_id)
    min_id = min(candidates)
    print(min_id)