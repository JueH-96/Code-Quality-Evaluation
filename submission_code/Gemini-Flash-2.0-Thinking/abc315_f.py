import math

def solve():
    n = int(input())
    coords = []
    for _ in range(n):
        x, y = map(int, input().split())
        coords.append((x, y))

    def distance(p1, p2):
        return math.sqrt((coords[p1][0] - coords[p2][0])**2 + (coords[p1][1] - coords[p2][1])**2)

    dp = [float('inf')] * n
    dp[0] = 0

    for i in range(1, n):
        for j in range(i):
            skipped_count = i - j - 1
            penalty = 0
            if skipped_count > 0:
                penalty = 2**(skipped_count - 1)

            dp[i] = min(dp[i], dp[j] + distance(j, i) + penalty)

    print(f"{dp[n-1]:.10f}")

solve()