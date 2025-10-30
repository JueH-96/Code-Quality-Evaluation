class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        from heapq import heappop, heappush
        import sys
        
        def dijkstra(graph, start, end):
            # Min-heap priority queue
            heap = []
            # Start with the source node
            heappush(heap, (0, start))
            # Distances to nodes initialized to infinity
            distances = [sys.maxsize] * n
            distances[start] = 0
            
            while heap:
                # Get the node with the smallest distance
                current_distance, current_node = heappop(heap)
                
                # If we reach the end node
                if current_node == end:
                    return current_distance
                
                # If a found distance is already better, skip processing
                if current_distance > distances[current_node]:
                    continue
                
                # Explore neighbors
                for neighbor, weight in graph[current_node]:
                    distance = current_distance + weight
                    
                    # Only consider this new path if it's better
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heappush(heap, (distance, neighbor))
            
            return distances[end]
        
        # Initial graph setup with existing roads
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append((i + 1, 1))  # road from i to i+1 with weight 1
        
        results = []
        
        # Process each query
        for u, v in queries:
            # Add the new road to the graph
            graph[u].append((v, 1))
            # Calculate the shortest path from 0 to n-1
            shortest_path_length = dijkstra(graph, 0, n - 1)
            results.append(shortest_path_length)
        
        return results