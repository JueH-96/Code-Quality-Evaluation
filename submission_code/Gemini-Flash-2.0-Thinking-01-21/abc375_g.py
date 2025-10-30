import collections
import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    
    def get_shortest_distance(road_indices_to_use):
        adj = collections.defaultdict(list)
        for index in road_indices_to_use:
            (u, v), w = edges[index]
            adj[u].append((v, w))
            adj[v].append((u, w))
            
        distances = {city: float('inf') for city in range(1, n + 1)}
        distances[1] = 0
        priority_queue = [(0, 1)]
        
        while priority_queue:
            current_distance, current_city = heapq.heappop(priority_queue)
            if current_distance > distances[current_city]:
                continue
            for neighbor, weight in adj[current_city]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    
        return distances[n]

    original_shortest_distance = get_shortest_distance(range(m))
    results = []
    for i in range(m):
        road_indices_without_i = [j for j in range(m) if j != i]
        shortest_distance_without_i = get_shortest_distance(road_indices_without_i)
        if original_shortest_distance != shortest_distance_without_i:
            results.append("Yes")
        else:
            results.append("No")
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()