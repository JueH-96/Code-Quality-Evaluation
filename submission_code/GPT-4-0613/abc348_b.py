import math

N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y, i+1))

distances = [[0]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i != j:
            x1, y1, _ = points[i]
            x2, y2, _ = points[j]
            distances[i][j] = math.sqrt((x1-x2)**2 + (y1-y2)**2)

for i in range(N):
    max_distance = max(distances[i])
    for j in range(N):
        if distances[i][j] == max_distance:
            print(points[j][2])
            break