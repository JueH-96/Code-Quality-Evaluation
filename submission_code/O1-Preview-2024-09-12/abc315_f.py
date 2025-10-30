# YOUR CODE HERE
import sys
import math
import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]

    N = n
    L = 20  # Maximum number of skipped checkpoints to consider
    INF = float('inf')

    dp = [ [INF] * (L+1) for _ in range(N) ]
    dp[0][0] = 0.0

    for i in range(1, N):
        for c in range(L+1):
            min_k = 1
            max_k = L+1
            for k in range(1, max_k+1):
                prev = i - k
                if prev < 0:
                    break
                skipped = k -1
                new_c = c + skipped
                if new_c > L:
                    continue
                x1, y1 = points[prev]
                x2, y2 = points[i]
                dist = math.hypot(x2 - x1, y2 - y1)
                if dp[prev][c] + dist < dp[i][new_c]:
                    dp[i][new_c] = dp[prev][c] + dist

    min_s = INF
    for c in range(L+1):
        total_dist = dp[N-1][c]
        if total_dist >= INF:
            continue
        if c == 0:
            penalty = 0.0
        else:
            penalty = 2.0 ** (c -1)
        s = total_dist + penalty
        if s < min_s:
            min_s = s

    print("%.20f" % min_s)

threading.Thread(target=main).start()