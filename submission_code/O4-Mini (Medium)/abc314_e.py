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
        C = float(next(it))
        P = int(next(it))
        S = [int(next(it)) for __ in range(P)]
        # split zeros and positives
        pos = [s for s in S if s > 0]
        cnt0 = P - len(pos)
        wheels.append((C, P, cnt0, pos))
    # dp[x] = expected cost to reach >= M from score x
    dp = [0.0] * (M + 1)
    # dp[M] = 0
    for x in range(M - 1, -1, -1):
        best = float('inf')
        # try each wheel
        for (C, P, cnt0, pos) in wheels:
            # denom = 1 - probability of 0-point
            # p0 = cnt0/P
            # denom = 1 - p0
            # denom must be > 0 since sum S > 0 ensures pos non-empty
            denom = 1.0 - (cnt0 / P)
            # sum of dp for positive outcomes
            sdp = 0.0
            # For each positive spin outcome
            for s in pos:
                nx = x + s
                if nx >= M:
                    nx = M
                sdp += dp[nx]
            # expected cost if picking this wheel:
            # E = (C + (1/P) * sdp) / denom
            val = (C + (sdp / P)) / denom
            if val < best:
                best = val
        dp[x] = best
    # output dp[0]
    # Print with sufficient precision
    print("{:.10f}".format(dp[0]))

if __name__ == "__main__":
    main()