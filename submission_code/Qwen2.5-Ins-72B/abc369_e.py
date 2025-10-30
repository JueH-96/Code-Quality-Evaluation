import sys
from heapq import heappop, heappush
from collections import defaultdict

def dijkstra(graph, start, end, required_bridges):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start, set())]
    
    while pq:
        d, u, used_bridges = heappop(pq)
        if u == end and required_bridges.issubset(used_bridges):
            return d
        for v, t, bridge in graph[u]:
            if bridge in required_bridges and bridge not in used_bridges:
                new_used_bridges = used_bridges | {bridge}
            else:
                new_used_bridges = used_bridges
            if d + t < dist[v]:
                dist[v] = d + t
                heappush(pq, (d + t, v, new_used_bridges))
    return float('inf')

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    index = 2
    
    graph = defaultdict(list)
    bridges = {}
    
    for i in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        t = int(data[index + 2])
        graph[u].append((v, t, i))
        graph[v].append((u, t, i))
        bridges[i] = (u, v, t)
        index += 3
    
    Q = int(data[index])
    index += 1
    
    results = []
    
    for _ in range(Q):
        K = int(data[index])
        index += 1
        required_bridges = set(int(data[index + i]) - 1 for i in range(K))
        index += K
        result = dijkstra(graph, 0, N - 1, required_bridges)
        results.append(result)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()