import sys
from bisect import bisect_left, bisect_right

def read_input():
    N, D = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    return N, D, points

def solve(N, D, points):
    x_coords, y_coords = zip(*points)
    x_coords, y_coords = sorted(x_coords), sorted(y_coords)
    count = 0
    for x, y in points:
        x_left = bisect_left(x_coords, x - D)
        x_right = bisect_right(x_coords, x + D)
        y_left = bisect_left(y_coords, y - D)
        y_right = bisect_right(y_coords, y + D)
        count += (x_right - x_left) * (y_right - y_left)
    return count

def main():
    N, D, points = read_input()
    print(solve(N, D, points))

if __name__ == "__main__":
    main()