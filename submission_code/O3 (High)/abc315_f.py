import sys
import math


def main() -> None:
    # read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    xs, ys = [], []
    for _ in range(N):
        xs.append(int(next(it)))
        ys.append(int(next(it)))

    # At most this many checkpoints can be skipped in an optimal solution.
    # (see analysis – 2^27 ~= 1.3e8 ≥ worst possible “all-edges” length)
    GLOBAL_T_MAX = 27
    T = min(N - 2, GLOBAL_T_MAX)      # actually reachable skip count

    INF = float('inf')
    # dp[i][t] = minimal travelled distance to reach checkpoint i (0-based),
    #            having skipped exactly t checkpoints so far.
    dp = [[INF] * (T + 1) for _ in range(N)]
    dp[0][0] = 0.0

    for i in range(N):
        xi, yi = xs[i], ys[i]
        for skipped_so_far in range(T + 1):
            cur = dp[i][skipped_so_far]
            if cur == INF:
                continue
            # we can skip up to remaining = T - skipped_so_far checkpoints
            remaining = T - skipped_so_far
            # the farthest next checkpoint index we are allowed to jump to
            # ( +1 because between i and next we skip (next-i-1) )
            limit = min(N - 1, i + remaining + 1)
            for nxt in range(i + 1, limit + 1):
                s = nxt - i - 1                  # checkpoints skipped in this hop
                new_skip = skipped_so_far + s
                dx = xs[nxt] - xi
                dy = ys[nxt] - yi
                dist = math.hypot(dx, dy)
                new_cost = cur + dist
                if new_cost < dp[nxt][new_skip]:
                    dp[nxt][new_skip] = new_cost

    best = INF
    for t in range(T + 1):
        if dp[N - 1][t] == INF:
            continue
        penalty = 0.0 if t == 0 else float(1 << (t - 1))  # 2^{t-1} for t>0
        best = min(best, dp[N - 1][t] + penalty)

    # output with full precision (Python's default repr of float is enough;
    # judge uses absolute/relative error tolerance)
    print("{:.15f}".format(best))


if __name__ == "__main__":
    main()