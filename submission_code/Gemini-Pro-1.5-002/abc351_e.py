# YOUR CODE HERE
def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    def dist(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if (dx + dy) % 2 != 0:
            return 0
        if dx < dy:
            return dy
        else:
            return dx
    
    total_dist = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total_dist += dist(points[i], points[j])
    
    print(total_dist)

solve()