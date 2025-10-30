from collections import defaultdict

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with initial roads
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append((i + 1, 1))
        
        # Function to perform Dijkstra's algorithm
        def dijkstra(start, end):
            distances = {node: float('inf') for node in graph}
            distances[start] = 0
            visited = set()
            queue = [(0, start)]
            
            while queue:
                current_distance, current_node = heapq.heappop(queue)
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                
                if current_node == end:
                    return current_distance
                
                for neighbor, weight in graph[current_node]:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(queue, (distance, neighbor))
            
            return distances[end]
        
        # Process each query
        answer = []
        for u, v in queries:
            graph[u].append((v, 1))
            answer.append(dijkstra(0, n - 1))
        
        return answer