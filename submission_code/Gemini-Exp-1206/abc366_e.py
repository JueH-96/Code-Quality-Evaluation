def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    count = 0
    for x in range(min_x - d, max_x + d + 1):
        for y in range(min_y - d, max_y + d + 1):
            total_dist = 0
            for px, py in points:
                total_dist += abs(x - px) + abs(y - py)
            if total_dist <= d:
                count += 1

    print(count)

solve()