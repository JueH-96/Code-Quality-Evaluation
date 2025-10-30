# YOUR CODE HERE
import sys

def solve():
    N, D = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    x_min, x_max = min(x for x, y in points), max(x for x, y in points)
    y_min, y_max = min(y for x, y in points), max(y for x, y in points)

    ans = 0
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            if sum(abs(x-xi) + abs(y-yi) for xi, yi in points) <= D:
                ans += 1

    print(ans)

solve()