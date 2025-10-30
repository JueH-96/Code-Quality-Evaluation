import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def solve():
    n = int(input())
    coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        coords.append((x, y))

    dp = [float('inf')] * n
    dp[0] = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            skipped = j - i - 1
            penalty = 0 if skipped == 0 else 2**(skipped - 1)
            dp[j] = min(dp[j], dp[i] + dist + penalty)

    print(dp[n-1])

solve()