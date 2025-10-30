import sys
import heapq

input = sys.stdin.read
data = input().split()

index = 0

def read_int():
    global index
    val = int(data[index])
    index += 1
    return val

N = read_int()
M = read_int()
Q = read_int()

roads = []
for _ in range(M):
    A = read_int() - 1
    B = read_int() - 1
    C = read_int()
    roads.append((A, B, C))

queries = []
for _ in range(Q):
    parts = read_int()
    if parts == 1:
        i = read_int() - 1
        queries.append((1, i))
    elif parts == 2:
        x = read_int() - 1
        y = read_int() - 1
        queries.append((2, x, y))

# To handle the road closures
active_roads = [True] * M

def dijkstra(source, target, active_roads):
    # Using a priority queue (min-heap)
    pq = []
    heapq.heappush(pq, (0, source))
    distances = {i: float('inf') for i in range(N)}
    distances[source] = 0
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node == target:
            return current_distance
        
        if current_distance > distances[current_node]:
            continue
        
        for i in range(M):
            if not active_roads[i]:
                continue
            A, B, C = roads[i]
            neighbor = B if current_node == A else (A if current_node == B else None)
            if neighbor is not None:
                new_distance = current_distance + C
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))
    
    return -1 if distances[target] == float('inf') else distances[target]

output = []
for query in queries:
    if query[0] == 1:
        # Close road i
        i = query[1]
        active_roads[i] = False
    elif query[0] == 2:
        # Find shortest path from x to y
        x = query[1]
        y = query[2]
        result = dijkstra(x, y, active_roads)
        output.append(str(result))

print("
".join(output))