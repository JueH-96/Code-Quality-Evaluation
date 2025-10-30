import sys
from heapq import heapify, heappop, heappush

def main():
    N = int(sys.stdin.readline())
    G = [[] for _ in range(N)]
    for _ in range(N-1):
        a, b, c = map(int, sys.stdin.readline().split())
        a -= 1
        b -= 1
        G[a].append((b, c))
        G[b].append((a, c))

    dist = [0]*N
    visited = [False]*N
    visited[0] = True
    heap = [(0, 0)]
    heapify(heap)

    while heap:
        v = heappop(heap)[1]
        for to, cost in G[v]:
            if visited[to]:
                continue
            dist[to] = dist[v] + cost
            visited[to] = True
            heappush(heap, (dist[to], to))

    print(sum(dist)*2 - max(dist))

if __name__ == "__main__":
    main()