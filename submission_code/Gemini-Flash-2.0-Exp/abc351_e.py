def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    def dist(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        if (x1 + y1) % 2 != (x2 + y2) % 2:
            return 0
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        return max(dx, dy)

    total_dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += dist(points[i], points[j])

    print(total_dist)

solve()