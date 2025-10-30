# YOUR CODE HERE
def solve():
    n, d = map(int, input().split())
    points = []
    for _ in range(n):
        points.append(list(map(int, input().split())))

    count = 0
    for x in range(-200, 201):
        for y in range(-200, 201):
            total_dist = 0
            for px, py in points:
                total_dist += abs(x - px) + abs(y - py)
            if total_dist <= d:
                count += 1
    print(count)

solve()