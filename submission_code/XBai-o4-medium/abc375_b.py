import math
import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    n = data[0]
    points = []
    idx = 1
    for _ in range(n):
        x = data[idx]
        y = data[idx + 1]
        points.append((x, y))
        idx += 2
    total = 0.0
    current_x, current_y = 0.0, 0.0
    for x, y in points:
        dx = x - current_x
        dy = y - current_y
        total += math.hypot(dx, dy)
        current_x, current_y = x, y
    dx = -current_x
    dy = -current_y
    total += math.hypot(dx, dy)
    print("{0:.15f}".format(total))

if __name__ == "__main__":
    main()