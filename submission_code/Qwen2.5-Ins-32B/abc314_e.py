import sys

def solve():
    from collections import defaultdict
    import heapq

    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, input().split())
        wheels.append((C, P, S))

    dp = [float('inf')] * (M + 1)
    dp[0] = 0
    queue = [(0, 0)]
    while queue:
        cost, points = heapq.heappop(queue)
        if cost > dp[points]:
            continue
        for C, P, S in wheels:
            for s in S:
                new_points = min(points + s, M)
                new_cost = cost + C
                if new_cost < dp[new_points]:
                    dp[new_points] = new_cost
                    heapq.heappush(queue, (new_cost, new_points))

    print(dp[M])

solve()