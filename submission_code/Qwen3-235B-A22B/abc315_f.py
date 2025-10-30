import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = []
    Y = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        X.append(x)
        Y.append(y)
        idx += 2

    K = 20  # steps back to consider
    max_c = 20  # maximum skipped checkpoints considered

    INF = float('inf')
    dp = [[INF] * (max_c + 1) for _ in range(N + 1)]
    dp[1][0] = 0.0

    for i in range(2, N + 1):
        start_j = max(1, i - K)
        for j in range(start_j, i):
            dx = X[i-1] - X[j-1]
            dy = Y[i-1] - Y[j-1]
            dist = math.hypot(dx, dy)
            skipped = i - j - 1
            if skipped > max_c:
                continue
            max_prev_c = max_c - skipped
            for c_prev in range(0, max_prev_c + 1):
                c_current = c_prev + skipped
                if dp[j][c_prev] + dist < dp[i][c_current]:
                    dp[i][c_current] = dp[j][c_prev] + dist

    min_s = INF
    for c in range(0, max_c + 1):
        if c == 0:
            penalty = 0.0
        else:
            penalty = 1 << (c - 1)
        if dp[N][c] < INF:
            total = dp[N][c] + penalty
            if total < min_s:
                min_s = total

    print("{0:.20f}".format(min_s))

if __name__ == '__main__':
    main()