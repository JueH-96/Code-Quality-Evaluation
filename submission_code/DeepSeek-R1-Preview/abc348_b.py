n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(n):
    max_sq = -1
    best = []
    for j in range(n):
        if i == j:
            continue
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        sq = dx * dx + dy * dy
        if sq > max_sq:
            max_sq = sq
            best = [j + 1]
        elif sq == max_sq:
            best.append(j + 1)
    print(min(best))