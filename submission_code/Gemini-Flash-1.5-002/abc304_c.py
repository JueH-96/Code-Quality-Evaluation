import math

def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    infected = [False] * n
    infected[0] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if infected[i]:
                for j in range(n):
                    if not infected[j]:
                        dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                        if dist <= d:
                            infected[j] = True
                            changed = True

    for i in range(n):
        print("Yes" if infected[i] else "No")

solve()