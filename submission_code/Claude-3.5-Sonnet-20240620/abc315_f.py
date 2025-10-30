# YOUR CODE HERE
import sys
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def solve():
    N = int(sys.stdin.readline())
    checkpoints = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    dp = [float('inf')] * N
    dp[0] = 0

    for i in range(1, N):
        x1, y1 = checkpoints[i]
        for j in range(i):
            x2, y2 = checkpoints[j]
            dist = distance(x1, y1, x2, y2)
            penalty = 0 if j == i-1 else 2**(i-j-2)
            dp[i] = min(dp[i], dp[j] + dist + penalty)

    print(f"{dp[N-1]:.20f}")

solve()