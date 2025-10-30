import sys
import threading
def main():
    import sys, math
    data = sys.stdin.read().split()
    N = int(data[0])
    coords = []
    idx = 1
    for i in range(N):
        x = float(data[idx]); y = float(data[idx+1])
        coords.append((x,y))
        idx += 2

    # Maximum skips to consider
    M_max = 30

    INF = float('inf')
    # dp[i][m] = minimal distance sum to reach checkpoint i with exactly m skips so far
    # i = 0..N-1, m = 0..M_max
    dp = [ [INF]*(M_max+1) for _ in range(N) ]
    dp[0][0] = 0.0

    for i in range(1, N):
        xi, yi = coords[i]
        # j runs from max(0, i-(M_max+1)) to i-1
        lo = i - (M_max+1)
        if lo < 0: lo = 0
        # For each possible predecessor j
        for j in range(lo, i):
            xj, yj = coords[j]
            s = i - j - 1  # number of skips in this jump
            if s > M_max:
                continue
            dx = xi - xj
            dy = yi - yj
            d = math.hypot(dx, dy)
            # Update dp[i][m] for all m>=s
            dj = dp[j]
            di = dp[i]
            # unroll m from s..M_max
            for m in range(s, M_max+1):
                prev = dj[m - s]
                if prev + d < di[m]:
                    di[m] = prev + d

    # Precompute penalties
    penalties = [0.0]*(M_max+1)
    for m in range(1, M_max+1):
        penalties[m] = float(1 << (m-1))

    ans = float('inf')
    last = dp[N-1]
    for m in range(0, M_max+1):
        val = last[m]
        if val == INF:
            continue
        pen = penalties[m]
        total = val + pen
        if total < ans:
            ans = total

    # Print with sufficient precision
    sys.stdout.write("{:.15f}
".format(ans))

if __name__ == "__main__":
    main()