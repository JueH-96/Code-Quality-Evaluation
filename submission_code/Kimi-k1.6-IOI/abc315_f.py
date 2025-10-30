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
    
    K = 30  # Maximum allowed skips
    INF = float('inf')
    dp = [[INF] * (K + 1) for _ in range(N)]
    dp[0][0] = 0.0  # Starting at checkpoint 0 with 0 skips
    
    for i in range(N):
        for c in range(K + 1):
            if dp[i][c] == INF:
                continue
            # Consider all j from i+1 to min(i + K + 1, N-1)
            max_j = min(i + K + 1, N - 1)
            for j in range(i + 1, max_j + 1):
                s_add = j - i - 1
                new_c = c + s_add
                if new_c > K:
                    continue
                dx = X[j] - X[i]
                dy = Y[j] - Y[i]
                dist = math.hypot(dx, dy)
                if dp[j][new_c] > dp[i][c] + dist:
                    dp[j][new_c] = dp[i][c] + dist
    
    min_s = INF
    for c in range(K + 1):
        if dp[N - 1][c] != INF:
            penalty = 2 ** (c - 1) if c > 0 else 0.0
            total = dp[N - 1][c] + penalty
            if total < min_s:
                min_s = total
    print("{0:.20f}".format(min_s))

if __name__ == "__main__":
    main()