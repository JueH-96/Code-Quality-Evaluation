import sys
import math

def main():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    coords = list(map(int, data[1:]))

    total_cost = 0.0
    prev_x, prev_y = 0, 0
    idx = 0

    for _ in range(n):
        x = coords[idx]
        y = coords[idx+1]
        total_cost += math.hypot(x - prev_x, y - prev_y)
        prev_x, prev_y = x, y
        idx += 2

    # return to origin
    total_cost += math.hypot(prev_x, prev_y)

    # print with high precision
    print("{:.15f}".format(total_cost))

if __name__ == "__main__":
    main()