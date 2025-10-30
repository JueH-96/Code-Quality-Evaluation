import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N, R, C = map(int, input().split())
    S = input().strip()

    # starts holds the prefix sums at generation start times.
    # prefix (0,0) at time 0 is an initial start.
    starts = set()
    px = 0
    py = 0
    starts.add((0, 0))

    res = []

    for ch in S:
        # move according to wind
        if ch == 'N':
            px -= 1
        elif ch == 'S':
            px += 1
        elif ch == 'W':
            py -= 1
        elif ch == 'E':
            py += 1

        # check if zero‐sum segment ending here exists
        # i.e. if current prefix equals any start prefix
        if (px, py) not in starts:
            # no zero‐sum path back to origin, so new smoke is generated here
            starts.add((px, py))

        # check if smoke reaches (R, C) at this half‐step:
        # need a start prefix (sx, sy) so that (px - sx, py - sy) == (R, C)
        # i.e. (sx, sy) == (px - R, py - C)
        need = (px - R, py - C)
        if need in starts:
            res.append('1')
        else:
            res.append('0')

    print(''.join(res))


if __name__ == "__main__":
    main()