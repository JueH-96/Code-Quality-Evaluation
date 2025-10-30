# YOUR CODE HERE
def dist(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx > dy:
        return dx
    else:
        return dy

N = int(input().strip())
points = []
for _ in range(N):
    x, y = map(int, input().strip().split())
    points.append((x, y))

total = 0
for i in range(N):
    for j in range(i+1, N):
        total += dist(points[i], points[j])

print(total)