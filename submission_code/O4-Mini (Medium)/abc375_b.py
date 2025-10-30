import sys
import math

def main():
    data = sys.stdin
    n_line = data.readline()
    if not n_line:
        return
    N = int(n_line)
    prev_x, prev_y = 0, 0
    total_cost = 0.0

    for _ in range(N):
        x_str, y_str = data.readline().split()
        x = int(x_str)
        y = int(y_str)
        # cost from previous point to current
        dx = x - prev_x
        dy = y - prev_y
        total_cost += math.hypot(dx, dy)
        prev_x, prev_y = x, y

    # cost from last point back to origin
    total_cost += math.hypot(prev_x, prev_y)

    # print with sufficient precision
    print("{:.10f}".format(total_cost))

# call main to execute
main()