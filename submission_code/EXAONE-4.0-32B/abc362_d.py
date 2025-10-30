import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    edges = data[2+n:2+n+3*m]
    
    graph = [[] for _ in range(n+1)]
    index = 0
    for _ in range(m):
        u = int(edges[index])
        v = int(edges[index+1])
        b = int(edges[index+2])
        index += 3
        graph[u].append((v, b))
        graph[v].append((u, b))
    
    INF = 10**18
    dist = [INF] * (n+1)
    dist[1] = A[0]
    
    heap = []
    heapq.heappush(heap, (dist[1], 1))
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, b in graph[u]:
            new_dist = d + b + A[v-1]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    res = []
    for i in range(2, n+1):
        res.append(str(dist[i]))
    print(" ".join(res))

if __name__ == "__main__":
    main()