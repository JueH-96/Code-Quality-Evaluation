# YOUR CODE HERE
import sys
from collections import deque

def min_dissatisfaction(N, K, X, T):
    dp = [float('inf')] * (N + 1)
    dp[0] = 0
    queue = deque()

    for i in range(1, N + 1):
        while queue and queue[0][0] < T[i-1]:
            queue.popleft()

        while len(queue) >= K:
            queue.popleft()

        if queue:
            last_ship, last_dp = queue[-1]
            dp[i] = min(dp[i], last_dp + T[i-1] - last_ship)

        ship_day = max(T[i-1], queue[-1][0] + X if queue else T[i-1])
        while queue and queue[-1][1] > dp[i-1] + ship_day - T[i-1]:
            queue.pop()
        queue.append((ship_day, dp[i-1] + ship_day - T[i-1]))

        dp[i] = min(dp[i], queue[-1][1])

    return dp[N]

# Read input
N, K, X = map(int, input().split())
T = list(map(int, input().split()))

# Solve and print the result
result = min_dissatisfaction(N, K, X, T)
print(result)