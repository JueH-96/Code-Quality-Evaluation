import heapq

def dijkstra(n, graph, weights):
    # Initialize distances with infinity
    dist = [float('inf')] * n
    dist[0] = weights[0]
    
    # Priority queue to store (distance, vertex)
    pq = [(weights[0], 0)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            new_dist = current_dist + weight + weights[v]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return dist

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    weights = list(map(int, data[2:n+2]))
    
    graph = [[] for _ in range(n)]
    idx = n + 2
    for _ in range(m):
        u = int(data[idx]) - 1
        v = int(data[idx + 1]) - 1
        w = int(data[idx + 2])
        graph[u].append((v, w))
        graph[v].append((u, w))
        idx += 3
    
    dist = dijkstra(n, graph, weights)
    
    print(' '.join(map(str, dist[1:])))

if __name__ == "__main__":
    main()