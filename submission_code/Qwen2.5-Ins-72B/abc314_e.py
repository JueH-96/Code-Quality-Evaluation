import sys
from heapq import heappop, heappush

def solve(N, M, wheels):
    # Initialize the DP table
    dp = [float('inf')] * (M + 1)
    dp[0] = 0

    # Priority queue to store the states
    pq = [(0, 0)]

    while pq:
        cost, points = heappop(pq)
        if points == M:
            break
        for i in range(N):
            c, p, s = wheels[i]
            expected_cost = c
            expected_points = sum(s) / p
            new_points = min(M, points + expected_points)
            new_cost = cost + expected_cost
            if new_cost < dp[int(new_points)]:
                dp[int(new_points)] = new_cost
                heappush(pq, (new_cost, int(new_points)))

    return dp[M]

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
wheels = []
index = 2
for _ in range(N):
    C = int(data[index])
    P = int(data[index + 1])
    S = list(map(int, data[index + 2:index + 2 + P]))
    wheels.append((C, P, S))
    index += 2 + P

# Solve the problem
result = solve(N, M, wheels)
print(result)