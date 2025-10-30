import heapq

# Read input
N, M, Q = map(int, input().split())
roads = []
graph = [[] for _ in range(N+1)]

for i in range(1, M+1):
    A, B, C = map(int, input().split())
    roads.append((A, B, C))
    graph[A].append((B, C, i))  # (destination, length, road_id)
    graph[B].append((A, C, i))  # Bidirectional road

closed_roads = set()

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        _, i = query
        closed_roads.add(i)
    
    else:  # query[0] == 2
        _, x, y = query
        
        # Dijkstra's algorithm
        distances = [float('infinity')] * (N+1)
        distances[x] = 0
        priority_queue = [(0, x)]
        visited = [False] * (N+1)
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if visited[current_node]:
                continue
                
            visited[current_node] = True
            
            if current_node == y:
                break
            
            for neighbor, weight, road_num in graph[current_node]:
                if road_num in closed_roads:
                    continue
                    
                if not visited[neighbor]:
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue, (distance, neighbor))
        
        if distances[y] == float('infinity'):
            print(-1)
        else:
            print(distances[y])