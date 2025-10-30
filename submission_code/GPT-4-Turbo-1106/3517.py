from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the graph with the initial roads
        graph = {i: {i + 1} for i in range(n - 1)}
        
        # Initialize the shortest distance to be the number of cities - 1
        shortest_distance = n - 1
        
        # Initialize the answer list
        answer = []
        
        # Process each query
        for u, v in queries:
            # Add the new road to the graph
            if u in graph:
                graph[u].add(v)
            else:
                graph[u] = {v}
            
            # Use BFS to find the shortest path from city 0 to city n - 1
            queue = [(0, 0)]  # (city, distance)
            visited = set()
            found = False
            
            while queue and not found:
                current_city, distance = queue.pop(0)
                if current_city == n - 1:
                    shortest_distance = distance
                    found = True
                    break
                if current_city in visited:
                    continue
                visited.add(current_city)
                for neighbor in graph.get(current_city, []):
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
            
            # Add the current shortest distance to the answer list
            answer.append(shortest_distance)
        
        return answer

# Example usage:
# sol = Solution()
# print(sol.shortestDistanceAfterQueries(5, [[2,4],[0,2],[0,4]]))  # Output: [3,2,1]
# print(sol.shortestDistanceAfterQueries(4, [[0,3],[0,2]]))  # Output: [1,1]