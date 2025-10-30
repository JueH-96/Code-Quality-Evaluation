n = int(input())
points = [None]  # Using 1-based indexing

for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(1, n + 1):
    distances = []
    for j in range(1, n + 1):
        if i == j:
            continue
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        dist_sq = dx ** 2 + dy ** 2
        distances.append((dist_sq, j))
    
    max_dist_sq = max(d[0] for d in distances)
    candidates = [d[1] for d in distances if d[0] == max_dist_sq]
    print(min(candidates))