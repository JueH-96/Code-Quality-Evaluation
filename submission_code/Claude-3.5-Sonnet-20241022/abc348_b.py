N = int(input())
points = []
for i in range(N):
    x, y = map(int, input().split())
    points.append((x, y, i+1))

def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

for i in range(N):
    max_dist = -1
    ans = float('inf')
    for j in range(N):
        if i == j:
            continue
        d = dist(points[i], points[j])
        if d > max_dist:
            max_dist = d
            ans = points[j][2]
        elif d == max_dist:
            ans = min(ans, points[j][2])
    print(ans)