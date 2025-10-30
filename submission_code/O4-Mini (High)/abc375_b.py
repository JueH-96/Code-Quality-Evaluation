import sys
import math

def main():
    input = sys.stdin.readline
    N = int(input())
    x_prev, y_prev = 0, 0
    total = 0.0

    for _ in range(N):
        x, y = map(int, input().split())
        # distance from previous point to current
        total += math.hypot(x - x_prev, y - y_prev)
        x_prev, y_prev = x, y

    # finally return to the origin
    total += math.hypot(x_prev, y_prev)

    # print with enough precision
    print("{:.10f}".format(total))

if __name__ == '__main__':
    main()