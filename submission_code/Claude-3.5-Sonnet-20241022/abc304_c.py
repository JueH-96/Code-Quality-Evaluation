import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

N, D = map(int, input().split())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x,y))

infected = [False] * N
infected[0] = True
changed = True

while changed:
    changed = False
    for i in range(N):
        if infected[i]:
            for j in range(N):
                if not infected[j] and distance(points[i][0], points[i][1], points[j][0], points[j][1]) <= D:
                    infected[j] = True
                    changed = True

for i in range(N):
    print("Yes" if infected[i] else "No")