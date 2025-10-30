import sys
import heapq

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    queries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    for query in queries:
        b, k = query
        distances = [(abs(a - b), i) for i, (a, _) in enumerate(points)]
        heapq.heapify(distances)
        closest = [heapq.heappop(distances)[1] for _ in range(k)]
        print(sorted([points[i][0] for i in closest])[k-1])

solve()