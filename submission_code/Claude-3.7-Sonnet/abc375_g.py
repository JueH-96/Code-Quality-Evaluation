import heapq

def dijkstra(graph, start, end, exclude_road=None):
    # Initialize distances with infinity
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
            
        if current_node == end:
            return current_distance
            
        for neighbor, weight, road_idx in graph[current_node]:
            if road_idx == exclude_road:
                continue
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return float('infinity')  # If end is unreachable

def solve():
    N, M = map(int, input().split())
    
    # Construct the graph
    graph = {i: [] for i in range(1, N + 1)}
    for i in range(M):
        a, b, c = map(int, input().split())
        graph[a].append((b, c, i))
        graph[b].append((a, c, i))
    
    shortest_with_all = dijkstra(graph, 1, N)
    
    for i in range(M):
        shortest_without_i = dijkstra(graph, 1, N, i)
        
        if shortest_with_all != shortest_without_i:
            print("Yes")
        else:
            print("No")

solve()