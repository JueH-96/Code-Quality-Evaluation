# N = int(input())
# points = [tuple(map(int, input().split())) for _ in range(N)]

from collections import defaultdict

def dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def solve(points):
    N = len(points)
    dp = defaultdict(int)
    for i in range(N):
        for j in range(i + 1, N):
            d = dist(points[i], points[j])
            dp[d] += 1
    ans = 0
    for d in dp:
        ans += dp[d] * (dp[d] - 1) // 2 * d
    return ans

N = int(input())
points = [tuple(map(int, input().split())) for _ in range(N)]
print(solve(points))