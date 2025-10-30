# YOUR CODE HERE
import sys
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            distance = current_dist + weight
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    return dist

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    index += 1
    edges = []
    for _ in range(N - 1):
        U = int(input[index]) - 1
        index += 1
        V = int(input[index]) - 1
        index += 1
        L = int(input[index])
        index += 1
        edges.append((U, V, L))
    
    graph = [[] for _ in range(N)]
    for U, V, L in edges:
        graph[U].append((V, L))
        graph[V].append((U, L))
    
    dist_from_0 = dijkstra(graph, 0)
    
    max_diameter = 0
    farthest_node = 0
    for i in range(N):
        if dist_from_0[i] > max_diameter:
            max_diameter = dist_from_0[i]
            farthest_node = i
    
    dist_from_farthest = dijkstra(graph, farthest_node)
    
    second_farthest_node = 0
    for i in range(N):
        if dist_from_farthest[i] > max_diameter:
            max_diameter = dist_from_farthest[i]
            second_farthest_node = i
    
    diameter = max_diameter
    
    result = [0] * N
    for K in range(1, N + 1):
        if K == 1:
            result[K - 1] = 2 * max(dist_from_farthest)
        elif K == 2:
            result[K - 1] = 2 * (max(dist_from_farthest) + max(dist_from_0))
        else:
            result[K - 1] = 2 * diameter
    
    for r in result:
        print(r)

if __name__ == "__main__":
    main()