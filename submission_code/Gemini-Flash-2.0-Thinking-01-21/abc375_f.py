import collections
import heapq

def solve():
    n, m, q = map(int, input().split())
    roads = []
    for i in range(m):
        u, v, w = map(int, input().split())
        roads.append(((u, v), w))
    
    queries = []
    for _ in range(q):
        query_parts = list(map(int, input().split()))
        queries.append(query_parts)
        
    is_road_closed = [False] * m
    
    road_indices_per_city = collections.defaultdict(list)
    for i in range(m):
        (u, v), w = roads[i]
        road_indices_per_city[u].append(i)
        road_indices_per_city[v].append(i)
        
    results = []
    for query in queries:
        query_type = query[0]
        if query_type == 1:
            road_index_to_close = query[1]
            is_road_closed[road_index_to_close-1] = True
        elif query_type == 2:
            start_city, end_city = query[1], query[2]
            distances = {city: float('inf') for city in range(1, n + 1)}
            distances[start_city] = 0
            pq = [(0, start_city)]
            
            while pq:
                current_distance, current_city = heapq.heappop(pq)
                if current_distance > distances[current_city]:
                    continue
                
                for road_index in road_indices_per_city[current_city]:
                    if not is_road_closed[road_index]:
                        (city1, city2), weight = roads[road_index]
                        neighbor_city = -1
                        if city1 == current_city:
                            neighbor_city = city2
                        elif city2 == current_city:
                            neighbor_city = city1
                        else:
                            continue
                            
                        if distances[current_city] + weight < distances[neighbor_city]:
                            distances[neighbor_city] = distances[current_city] + weight
                            heapq.heappush(pq, (distances[neighbor_city], neighbor_city))
                            
            shortest_distance = distances[end_city]
            if shortest_distance == float('inf'):
                results.append("-1")
            else:
                results.append(str(shortest_distance))
                
    print('
'.join(results))

if __name__ == '__main__':
    solve()