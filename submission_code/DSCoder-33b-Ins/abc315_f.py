import sys
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def solve():
    n = int(input())
    points = [tuple(map(int, input().split())) for _ in range(n)]
    dp = [0] * n
    dp[1] = distance(points[0], points[1])
    for i in range(2, n):
        dp[i] = min(dp[j] + distance(points[j], points[i]) for j in range(i - 1))
    print(dp[-1])

solve()