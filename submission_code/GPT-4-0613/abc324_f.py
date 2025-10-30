import sys
import heapq
from collections import defaultdict

def main():
    n, m = map(int, sys.stdin.readline().split())
    edges = defaultdict(list)
    for _ in range(m):
        u, v, b, c = map(int, sys.stdin.readline().split())
        edges[u].append((v, b, c))

    dp = [0] * (n + 1)
    heap = [(0, 1)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost < dp[node]:
            continue
        for v, b, c in edges[node]:
            new_cost = max(dp[node], b / c)
            if new_cost > dp[v]:
                dp[v] = new_cost
                heapq.heappush(heap, (-new_cost, v))

    print(dp[n])

if __name__ == "__main__":
    main()