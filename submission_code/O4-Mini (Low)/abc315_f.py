import sys
import threading

def main():
    import sys, math

    data = sys.stdin.read().split()
    N = int(data[0])
    coords = []
    idx = 1
    for i in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        idx += 2
        coords.append((x, y))

    # We only need to consider skipping up to Cmax checkpoints,
    # where penalty = 2^(C-1) becomes prohibitively large.
    # Rough upper‐bound on the total travel distance is ~1e8,
    # so once 2^(C-1) >> 1e8 it's never optimal.
    # 2^27 ~ 1.34e8, so let's take Cmax = 30.
    Cmax = 30

    INF = float('inf')
    # dp[i][c] = minimum travel distance to reach checkpoint i
    # having skipped exactly c checkpoints so far.
    dp = [[INF] * (Cmax + 1) for _ in range(N)]
    dp[0][0] = 0.0

    for i in range(N - 1):
        xi, yi = coords[i]
        # for each number of skips so far
        for c_old in range(Cmax + 1):
            base = dp[i][c_old]
            if base == INF:
                continue
            # try jumping to j > i, skipping k = j-i-1 intermediate
            # only need j up to i+Cmax+1, since skip count would exceed otherwise
            jmax = min(N - 1, i + Cmax + 1)
            for j in range(i + 1, jmax + 1):
                k = j - i - 1
                c_new = c_old + k
                if c_new > Cmax:
                    break
                xj, yj = coords[j]
                dx = xi - xj
                dy = yi - yj
                dist = math.hypot(dx, dy)
                newv = base + dist
                if newv < dp[j][c_new]:
                    dp[j][c_new] = newv

    # Now compute the answer by adding the penalty for each c
    best = INF
    for c in range(Cmax + 1):
        travel = dp[N-1][c]
        if travel == INF:
            continue
        if c == 0:
            penalty = 0.0
        else:
            penalty = float(1 << (c - 1))
        total = travel + penalty
        if total < best:
            best = total

    # Print with sufficient precision
    sys.stdout.write("{:.15f}
".format(best))


if __name__ == "__main__":
    main()