import heapq
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        a = int(next(it)); b = int(next(it))
        graph[a].append(b)
        graph[b].append(a)
    
    guards = []
    for _ in range(k):
        p = int(next(it)); h = int(next(it))
        guards.append((p, h))
    
    INF = 10**18
    d = [INF] * (n+1)
    q = []
    
    for p, h in guards:
        val = -h
        if val < d[p]:
            d[p] = val
            heapq.heappush(q, (val, p))
    
    while q:
        dist_u, u = heapq.heappop(q)
        if dist_u != d[u]:
            continue
        for v in graph[u]:
            candidate = dist_u + 1
            if candidate < d[v]:
                d[v] = candidate
                heapq.heappush(q, (candidate, v))
    
    guarded_vertices = []
    for i in range(1, n+1):
        if d[i] <= 0:
            guarded_vertices.append(i)
    
    guarded_vertices.sort()
    print(len(guarded_vertices))
    if guarded_vertices:
        print(" ".join(map(str, guarded_vertices)))
    else:
        print()

if __name__ == "__main__":
    main()