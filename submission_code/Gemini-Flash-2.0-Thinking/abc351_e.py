def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    total_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]

            dx = x2 - x1
            dy = y2 - y1

            if dx % 2 == dy % 2:
                distance = max(abs(dx), abs(dy))
                total_distance += distance
            else:
                total_distance += 0

    print(total_distance)

solve()