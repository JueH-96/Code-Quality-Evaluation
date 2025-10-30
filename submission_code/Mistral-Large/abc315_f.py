import math
import sys
input = sys.stdin.read

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def min_cost(N, points):
    # Initialize DP table
    dp = [float('inf')] * (N + 1)
    dp[0] = 0

    # Precompute distances
    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            dist[i][j] = distance(points[i][0], points[i][1], points[j][0], points[j][1])

    # Fill DP table
    for i in range(1, N):
        for j in range(i):
            penalty = 0 if i == j + 1 else (1 << (i - j - 2))
            dp[i] = min(dp[i], dp[j] + dist[j][i] + penalty)

    return dp[N - 1]

def main():
    data = input().split()
    N = int(data[0])
    points = []
    index = 1
    for i in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2

    result = min_cost(N, points)
    print(result)

if __name__ == "__main__":
    main()