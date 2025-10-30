import collections
import heapq

def solve():
    n, m = map(int, input().split())
    roads = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        roads.append(((u, v), w))
    
    def get_shortest_distance(current_roads):
        adj = collections.defaultdict(list)
        for i in range(len(current_roads)):
            (u, v), w = current_roads[i]
            adj[u].append((v, w))
            adj[v].append((u, w))
            
        dist = {city: float('inf') for city in range(1, n + 1)}
        dist[1] = 0
        pq = [(0, 1)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, weight in adj[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
                    
        return dist[n]

    shortest_dist_all_roads = get_shortest_distance(roads)
    
    results = []
    for i in range(m):
        road_to_remove = roads[i]
        roads_without_i = []
        for j in range(m):
            if i != j:
                roads_without_i.append(roads[j])
        
        shortest_dist_without_road_i = get_shortest_distance(roads_without_i)
        
        if shortest_dist_all_roads == float('inf') and shortest_dist_without_road_i == float('inf'):
            different = False
        elif shortest_dist_all_roads == float('inf') or shortest_dist_without_road_i == float('inf'):
            different = True
        else:
            different = shortest_dist_all_roads != shortest_dist_without_road_i
            
        if different:
            results.append("Yes")
        else:
            results.append("No")
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()