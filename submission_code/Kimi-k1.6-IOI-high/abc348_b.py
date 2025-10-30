n = int(input())
points = [ (0, 0) ]  # Dummy entry at index 0
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

for i in range(1, n + 1):
    max_sq = -1
    best_id = 0
    for j in range(1, n + 1):
        if i == j:
            continue
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        sq = dx ** 2 + dy ** 2
        if sq > max_sq:
            max_sq = sq
            best_id = j
        elif sq == max_sq:
            if j < best_id:
                best_id = j
    print(best_id)