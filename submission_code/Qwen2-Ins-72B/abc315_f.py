from math import sqrt
from bisect import bisect_left

def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def solve(points):
    N = len(points)
    dp = [float('inf')] * N
    dp[0] = 0
    for i in range(1, N):
        for j in range(i):
            dp[i] = min(dp[i], dp[j] + dist(points[i], points[j]))
        for j in range(i):
            for k in range(j):
                dp[i] = min(dp[i], dp[k] + dist(points[i], points[j]) + 2 ** (i - k - 1))
    return dp[-1]

def main():
    N = int(input())
    points = [tuple(map(int, input().split())) for _ in range(N)]
    points.sort()
    print(solve(points))

if __name__ == "__main__":
    main()