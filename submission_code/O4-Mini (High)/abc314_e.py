import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    wheels = []
    for _ in range(N):
        C = int(next(it))
        P = int(next(it))
        S = [int(next(it)) for __ in range(P)]
        # collect only positive rewards
        pos = [s for s in S if s > 0]
        k = len(pos)
        # P*C used in numerator
        wheels.append((C * P, P, pos, k))
    # dp[x] = minimal expected cost from state x (<M) to reach >=M
    # dp[M] = 0
    dp = [0.0] * (M + 1)
    # compute dp[x] for x = M-1 down to 0
    for x in range(M - 1, -1, -1):
        best = float('inf')
        # try each wheel
        for (CP, P, pos, k) in wheels:
            # sum of dp at successors (for positive s)
            sdp = 0.0
            # accumulate dp[x+s] clamped to M
            # if pos is empty, k==0 can't happen by problem statement
            for s in pos:
                nxt = x + s
                if nxt >= M:
                    # terminal
                    # dp[M] = 0
                    continue
                sdp += dp[nxt]
            # expected cost using this wheel:
            # dp_i = (C*P + sum dp[x+s]) / k
            val = (CP + sdp) / k
            if val < best:
                best = val
        dp[x] = best
    # output dp[0]
    # print with sufficient precision
    sys.stdout.write("{:.10f}
".format(dp[0]))

if __name__ == "__main__":
    main()