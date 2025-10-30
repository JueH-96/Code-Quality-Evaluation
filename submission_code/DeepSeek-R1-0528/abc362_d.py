import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    edges = data[2+n:2+n+3*m]
    
    graph = [[] for _ in range(n)]
    
    for i in range(m):
        u = int(edges[3*i])
        v = int(edges[3*i+1])
        b = int(edges[3*i+2])
        u -= 1
        v -= 1
        graph[u].append((v, b + A[v]))
        graph[v].append((u, b + A[u]))
    
    INF = 10**18
    dist = [INF] * n
    dist[0] = A[0]
    heap = [(A[0], 0)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            new_d = d + w
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))
                
    print(" ".join(str(x) for x in dist[1:]))

if __name__ == "__main__":
    main()