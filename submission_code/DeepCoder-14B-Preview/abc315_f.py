import math

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    points = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        points.append((x, y))
        idx += 2
    
    # Precompute the distance matrix
    dist = [[0.0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            dx = points[i - 1][0] - points[j - 1][0]
            dy = points[i - 1][1] - points[j - 1][1]
            d = math.hypot(dx, dy)
            dist[i][j] = d
            dist[j][i] = d  # Since distance is symmetric
    
    # Initialize DP table
    INF = float('inf')
    dp = [[INF] * (N + 1) for _ in range(N + 1)]
    dp[1][1] = 0.0  # Starting at checkpoint 1 with 1 checkpoint used
    
    for k in range(1, N):
        for i in range(1, N + 1):
            if dp[k][i] == INF:
                continue
            for j in range(i + 1, N + 1):
                if dp[k + 1][j] > dp[k][i] + dist[i][j]:
                    dp[k + 1][j] = dp[k][i] + dist[i][j]
    
    min_s = INF
    for m in range(2, N + 1):
        current_sum = dp[N][m]
        if current_sum == INF:
            continue
        if m < N:
            C = N - m
            penalty = 2 ** (C - 1)
        else:
            penalty = 0
        total = current_sum + penalty
        if total < min_s:
            min_s = total
    
    # Print with sufficient precision
    print("{0:.10f}".format(min_s))

if __name__ == '__main__':
    main()