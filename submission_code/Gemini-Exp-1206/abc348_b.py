def solve():
    n = int(input())
    points = []
    for _ in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    def distance(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    for i in range(n):
        max_dist = -1
        farthest_point_id = -1
        for j in range(n):
            if i == j:
                continue
            dist = distance(points[i], points[j])
            if dist > max_dist:
                max_dist = dist
                farthest_point_id = j + 1
            elif dist == max_dist:
                farthest_point_id = min(farthest_point_id, j + 1)
        print(farthest_point_id)

solve()