import sys

def solve():
    N, D = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    count = 0
    for x in range(-D, D+1):
        for y in range(-D, D+1):
            if (x, y) in points:
                count += sum(abs(x-point[0])+abs(y-point[1]) for point in points)
    print(count)

solve()