import heapq

def has_empty_intersection(range1, range2):
    l1, r1 = range1
    l2, r2 = range2
    return r1 < l2 or r2 < l1

def dijkstra(graph, start, end, weights):
    # Initialize distances with infinity for all vertices
    distances = {vertex: float('infinity') for vertex in range(1, len(graph) + 1)}
    distances[start] = weights[start - 1]  # Weight of the starting vertex
    
    # Priority queue to pick the vertex with the minimum distance
    priority_queue = [(weights[start - 1], start)]
    
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        
        # If the current distance is greater than the known distance, skip
        if current_distance > distances[current_vertex]:
            continue
        
        # If we've reached the end vertex, return the distance
        if current_vertex == end:
            return current_distance
        
        # Check all neighboring vertices
        for neighbor in graph[current_vertex]:
            # The distance is the current distance (which already includes the
            # weight of the current vertex) plus the weight of the neighbor
            distance = current_distance + weights[neighbor - 1]
            
            # If this path is shorter than known, update and add to the queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # If we can't reach the end vertex, return -1
    return -1

# Parse input
N = int(input())
weights = list(map(int, input().split()))
ranges = []
for _ in range(N):
    L, R = map(int, input().split())
    ranges.append((L, R))

# Build the graph
graph = {i: [] for i in range(1, N + 1)}
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        if has_empty_intersection(ranges[i - 1], ranges[j - 1]):
            graph[i].append(j)
            graph[j].append(i)

# Process queries
Q = int(input())
for _ in range(Q):
    s, t = map(int, input().split())
    result = dijkstra(graph, s, t, weights)
    print(result)