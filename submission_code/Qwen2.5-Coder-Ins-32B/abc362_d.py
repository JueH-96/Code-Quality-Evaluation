# YOUR CODE HERE
import sys
import heapq

def dijkstra(graph, start, vertex_weights):
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = vertex_weights[start]
    priority_queue = [(vertex_weights[start], start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_vertex]:
            continue
        
        for neighbor, edge_weight in graph[current_vertex]:
            distance = current_distance + edge_weight + vertex_weights[neighbor]
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    
    A = list(map(int, input[index:index + N]))
    index += N
    
    graph = [[] for _ in range(N)]
    
    for _ in range(M):
        U = int(input[index]) - 1
        V = int(input[index + 1]) - 1
        B = int(input[index + 2])
        index += 3
        
        graph[U].append((V, B))
        graph[V].append((U, B))
    
    distances = dijkstra(graph, 0, A)
    result = [str(distances[i]) for i in range(1, N)]
    print(" ".join(result))

if __name__ == "__main__":
    main()