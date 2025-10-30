import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    X = int(next(it))
    graph = [[] for _ in range(2 * n)]
    
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        u0 = 2 * (u - 1)
        v0 = 2 * (v - 1)
        u1 = u0 + 1
        v1 = v0 + 1
        graph[u0].append((v0, 1))
        graph[v1].append((u1, 1))
    
    for i in range(n):
        idx0 = 2 * i
        idx1 = 2 * i + 1
        graph[idx0].append((idx1, X))
        graph[idx1].append((idx0, X))
    
    INF = 10**18
    dist = [INF] * (2 * n)
    start = 0
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        for neighbor, weight in graph[node]:
            nd = d + weight
            if nd < dist[neighbor]:
                dist[neighbor] = nd
                heapq.heappush(heap, (nd, neighbor))
    
    target0 = 2 * (n - 1)
    target1 = 2 * (n - 1) + 1
    ans = min(dist[target0], dist[target1])
    print(ans)

if __name__ == "__main__":
    main()