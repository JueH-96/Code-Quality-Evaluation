def solve():
    n = int(input())
    points = []
    for i in range(n):
        x, y = map(int, input().split())
        points.append((x, y))

    for i in range(n):
        max_dist = -1
        farthest_point = -1
        for j in range(n):
            if i != j:
                dist = ((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)**0.5
                if dist > max_dist:
                    max_dist = dist
                    farthest_point = j + 1
                elif dist == max_dist:
                    farthest_point = min(farthest_point, j + 1)
        print(farthest_point)

solve()