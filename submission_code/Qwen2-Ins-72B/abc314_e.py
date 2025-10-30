from heapq import heappush, heappop
from math import inf

def solve(N, M, wheels):
    dp = [0] + [inf] * M
    pq = [(-wheels[i][0][1], 0, i) for i in range(N)]
    heapq.heapify(pq)
    while pq:
        cost, points, wheel = heappop(pq)
        if dp[points] < cost:
            continue
        for next_points, next_cost in wheels[wheel][1:]:
            next_points += points
            if next_points > M:
                next_points = M
            new_cost = cost + next_cost
            if dp[next_points] > new_cost:
                dp[next_points] = new_cost
                heappush(pq, (-new_cost, next_points, wheel))
    return dp[M]

def main():
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, input().split())
        expected_value = sum(S) / P
        wheel = []
        for s in S:
            wheel.append((s, C / (s + expected_value)))
        wheel.sort(reverse=True)
        wheels.append((expected_value, wheel))
    print(solve(N, M, wheels))

if __name__ == "__main__":
    main()