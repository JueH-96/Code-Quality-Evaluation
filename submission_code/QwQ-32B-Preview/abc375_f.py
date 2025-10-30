import sys
import heapq

def dijkstra(x, y, graph):
    dist = [float('inf')] * (N + 1)
    dist[x] = 0
    pq = [(0, x)]
    visited = [False] * (N + 1)
    
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == y:
            return d
        for v, c in graph[u].items():
            if not visited[v]:
                new_dist = dist[u] + c
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    return -1

def main():
    global N, M, Q, roads, queries
    input = sys.stdin.read().splitlines()
    NMQ = list(map(int, input[0].split()))
    global N, M, Q
    N = NMQ[0]
    M = NMQ[1]
    Q = NMQ[2]
    
    roads = []
    for i in range(1, M+1):
        A, B, C = map(int, input[i].split())
        roads.append((A, B, C))
    
    queries = []
    for i in range(M+1, M+1+Q):
        query = list(map(int, input[i].split()))
        queries.append(query)
    
    # Build graph as a list of dictionaries
    graph = [{} for _ in range(N + 1)]
    for A, B, C in roads:
        graph[A][B] = C
        graph[B][A] = C
    
    for query in queries:
        if query[0] == 1:
            # Close road i
            i = query[1]
            A, B, C = roads[i-1]
            del graph[A][B]
            del graph[B][A]
        else:
            # Find shortest path from x to y
            x, y = query[1], query[2]
            print(dijkstra(x, y, graph))

if __name__ == "__main__":
    main()