import sys
import math

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return

    # Read coordinates, 1-based
    X = [0] * (N+1)
    Y = [0] * (N+1)
    for i in range(1, N+1):
        X[i] = int(next(it))
        Y[i] = int(next(it))

    # We only need to consider skipping up to K checkpoints,
    # because penalty 2^{C-1} grows so fast that beyond ~30 skips
    # it's never optimal (max path‐length sum is < 1.5e8).
    K = N-2
    if K > 30:
        K = 30

    inf = float('inf')
    # dp[i][c] = minimal travel distance from 1 to i skipping exactly c checkpoints
    dp = [[inf] * (K+1) for _ in range(N+1)]
    dp[1][0] = 0.0

    sqrt = math.sqrt
    Xloc = X; Yloc = Y; dploc = dp

    for i in range(2, N+1):
        dp_i = dploc[i]
        xi = Xloc[i]; yi = Yloc[i]
        # maximum s we can skip in the last jump is i-2 (can't skip 1 or i)
        max_s = i-2
        if max_s > K:
            max_s = K

        # Try jumping from j = i-s-1 to i, which skips exactly s nodes
        for s in range(max_s+1):
            j = i - s - 1
            dx = xi - Xloc[j]
            dy = yi - Yloc[j]
            d = sqrt(dx*dx + dy*dy)
            dp_j = dploc[j]
            # update dp[i][c] for c = s..K using dp[j][c-s]
            # dp[i][c] = min(dp[i][c], dp[j][c-s] + d)
            # We unroll this as a small loop:
            for c in range(s, K+1):
                val = dp_j[c-s] + d
                if val < dp_i[c]:
                    dp_i[c] = val

    # Now consider the penalty: for c>0 penalty = 2^{c-1}, for c=0 penalty=0
    ans = dp[N][0]  # no skips => no penalty
    for c in range(1, K+1):
        dnc = dp[N][c]
        if dnc == inf:
            continue
        penalty = 1 << (c-1)
        total = dnc + penalty
        if total < ans:
            ans = total

    # Print with enough precision
    sys.stdout.write("{:.10f}
".format(ans))

main()