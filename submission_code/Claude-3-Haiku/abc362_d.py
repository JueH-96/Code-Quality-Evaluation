from collections import defaultdict
import heapq

def dijkstra(graph, start, end, vertex_weights):
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = vertex_weights[start - 1]
    heap = [(distances[start], start)]
    
    while heap:
        dist, u = heapq.heappop(heap)
        
        if u == end:
            return dist
        
        if dist > distances[u]:
            continue
        
        for v, weight in graph[u]:
            new_dist = dist + vertex_weights[v - 1] + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return float('inf')

def solve():
    N, M = map(int, input().split())
    vertex_weights = list(map(int, input().split()))
    graph = defaultdict(list)
    
    for _ in range(M):
        u, v, weight = map(int, input().split())
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    answers = []
    for i in range(2, N + 1):
        min_weight = dijkstra(graph, 1, i, vertex_weights)
        answers.append(str(min_weight))
    
    print(" ".join(answers))

solve()