import sys
from heapq import heappush, heappop
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    edges = defaultdict(list)
    for _ in range(N-1):
        u, v, l = map(int, sys.stdin.readline().split())
        edges[u].append((v, l))
        edges[v].append((u, l))

    dist = [0]*(N+1)
    visited = [False]*(N+1)
    heap = [(0, 1)]
    while heap:
        d, v = heappop(heap)
        if visited[v]:
            continue
        visited[v] = True
        dist[v] = d
        for to, cost in edges[v]:
            if visited[to]:
                continue
            heappush(heap, (d+cost, to))

    dist.sort(reverse=True)
    ans = [0]*N
    ans[0] = dist[0] + dist[1]
    for i in range(1, N):
        ans[i] = max(ans[i-1], dist[i] + dist[i+1] + dist[0])

    print('
'.join(map(str, ans)))

if __name__ == "__main__":
    main()