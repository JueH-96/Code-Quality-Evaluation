import sys
import heapq

def dijkstra(graph, start, closed_roads):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            if (current_node, neighbor) in closed_roads or (neighbor, current_node) in closed_roads:
                continue  # Skip closed roads

            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    first_line = data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    Q = int(first_line[2])
    
    graph = {i: [] for i in range(1, N + 1)}
    roads = []
    
    for i in range(1, M + 1):
        A, B, C = map(int, data[i].split())
        graph[A].append((B, C))
        graph[B].append((A, C))
        roads.append((A, B, C))
    
    closed_roads = set()
    output = []
    
    for i in range(M + 1, M + 1 + Q):
        query = list(map(int, data[i].split()))
        if query[0] == 1:
            road_index = query[1] - 1
            A, B, C = roads[road_index]
            closed_roads.add((A, B))
        elif query[0] == 2:
            x, y = query[1], query[2]
            distances = dijkstra(graph, x, closed_roads)
            result = distances[y] if distances[y] != float('inf') else -1
            output.append(result)
    
    print('
'.join(map(str, output)))

if __name__ == "__main__":
    main()