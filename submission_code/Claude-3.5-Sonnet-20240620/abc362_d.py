# YOUR CODE HERE
import heapq

def dijkstra(graph, start):
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = graph[start]['weight']
    pq = [(distances[start], start)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, weight in graph[current_vertex]['edges'].items():
            distance = current_distance + weight + graph[neighbor]['weight']
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

N, M = map(int, input().split())
A = list(map(int, input().split()))

graph = {i+1: {'weight': A[i], 'edges': {}} for i in range(N)}

for _ in range(M):
    U, V, B = map(int, input().split())
    graph[U]['edges'][V] = B
    graph[V]['edges'][U] = B

distances = dijkstra(graph, 1)

print(*distances[2:])