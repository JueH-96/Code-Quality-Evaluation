# YOUR CODE HERE
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = graph[start][0]
    pq = [(dist[start], start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph[u][1:]:
            if dist[u] + w + graph[v][0] < dist[v]:
                dist[v] = dist[u] + w + graph[v][0]
                heapq.heappush(pq, (dist[v], v))
    return dist

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:N+2]))
    edges = list(zip(map(int, data[N+2::3]), map(int, data[N+3::3]), map(int, data[N+4::3])))
    
    graph = [[A[i]] for i in range(N)]
    for u, v, b in edges:
        graph[u-1].append((v-1, b))
        graph[v-1].append((u-1, b))
    
    dist = dijkstra(graph, 0)
    print(' '.join(map(str, dist[1:])))

if __name__ == "__main__":
    main()