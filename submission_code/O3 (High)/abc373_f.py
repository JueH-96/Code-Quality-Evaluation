import sys
from collections import deque
from math import inf

INF_NEG = -10 ** 30


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    W = int(next(it))
    wv = [(int(next(it)), int(next(it))) for _ in range(N)]

    dp = [INF_NEG] * (W + 1)
    dp[0] = 0

    # utility functions for convex-hull optimisation
    def intersect_x(l1, l2):
        m1, b1 = l1
        m2, b2 = l2
        return (b2 - b1) / (m1 - m2)

    for w, v in wv:
        new_dp = [INF_NEG] * (W + 1)
        # process every residue modulo w separately
        for r in range(w):
            # list of weights having this residue:  r + t*w  (t = 0..t_max)
            t_max = (W - r) // w
            hull = []                # lines kept in increasing slope order
            ptr = 0                  # pointer to best line for current t
            for t in range(t_max + 1):
                weight = r + t * w
                if dp[weight] > INF_NEG:
                    m = 2 * t
                    b = dp[weight] - t * v - t * t
                    # add new line (m, b) to convex hull
                    while len(hull) >= 2 and intersect_x(hull[-2], hull[-1]) >= intersect_x(hull[-1], (m, b)):
                        hull.pop()
                    hull.append((m, b))
                if not hull:
                    continue
                # advance pointer while next line gives not worse value
                while ptr + 1 < len(hull) and hull[ptr][0] * t + hull[ptr][1] <= hull[ptr + 1][0] * t + hull[ptr + 1][1]:
                    ptr += 1
                best = hull[ptr][0] * t + hull[ptr][1]
                new_dp[weight] = t * v - t * t + best
        dp = new_dp

    print(max(dp))


if __name__ == "__main__":
    main()