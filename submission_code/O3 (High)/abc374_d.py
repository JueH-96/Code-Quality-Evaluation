import sys
import math


def hypot(p, q):
    """Euclidean distance between two points p and q"""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = float(next(it))  # speed when NOT emitting
    T = float(next(it))  # speed when emitting
    # store the two endpoints of every segment
    endpoints = []                 # endpoints[i][0 or 1] -> (x, y)
    seg_time = []                  # time to print segment i (independent of direction)
    for _ in range(N):
        a = float(next(it)); b = float(next(it))
        c = float(next(it)); d = float(next(it))
        endpoints.append(((a, b), (c, d)))
        seg_len = math.hypot(a - c, b - d)
        seg_time.append(seg_len / T)

    INF = float('inf')
    # dp[mask][i][d]  (mask: printed segments, i: last printed segment,
    # d: 0 or 1 meaning we STARTED that segment from endpoint d and ended at 1-d)
    size_mask = 1 << N
    dp = [[[INF, INF] for _ in range(N)] for _ in range(size_mask)]

    # initial states: go from origin to chosen starting endpoint, then print that segment
    origin = (0.0, 0.0)
    for i in range(N):
        for d in (0, 1):
            start_pt = endpoints[i][d]
            travel = hypot(origin, start_pt) / S
            mask = 1 << i
            dp[mask][i][d] = travel + seg_time[i]

    # iterate over masks
    for mask in range(size_mask):
        for i in range(N):
            for d in (0, 1):
                cur_time = dp[mask][i][d]
                if cur_time == INF:
                    continue
                # current position is the endpoint we finished at
                cur_pos = endpoints[i][1 - d]
                # try to print an unprinted segment next
                for k in range(N):
                    if mask & (1 << k):
                        continue  # already printed
                    for nd in (0, 1):  # choose starting endpoint for segment k
                        next_start = endpoints[k][nd]
                        move = hypot(cur_pos, next_start) / S
                        nxt_mask = mask | (1 << k)
                        new_time = cur_time + move + seg_time[k]
                        if new_time < dp[nxt_mask][k][nd]:
                            dp[nxt_mask][k][nd] = new_time

    full_mask = (1 << N) - 1
    answer = INF
    for i in range(N):
        for d in (0, 1):
            answer = min(answer, dp[full_mask][i][d])

    # print with plenty of precision
    print("{:.15f}".format(answer))


if __name__ == "__main__":
    main()