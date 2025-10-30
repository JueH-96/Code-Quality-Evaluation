from typing import List
from collections import deque

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Create an adjacency list representing our graph.
        # Initially, for every 0 <= i < n - 1, we have a road from i to i+1.
        graph = [[] for _ in range(n)]
        for i in range(n - 1):
            graph[i].append(i + 1)
        
        # This list will hold the shortest path length from 0 to n - 1 after each query.
        answer = []
        
        # Process each query by adding the new road and then performing a BFS.
        for u, v in queries:
            # Add the new road from city u to city v.
            graph[u].append(v)
            
            # Perform a BFS to find the shortest distance from city 0 to city n - 1.
            dist = [-1] * n  # Initialize distances as -1 (unvisited)
            dist[0] = 0    # Starting point with distance 0
            queue = deque([0])
            
            while queue:
                current = queue.popleft()
                if current == n - 1:
                    # We reached the destination; no need to search further.
                    break
                for neighbor in graph[current]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[current] + 1
                        queue.append(neighbor)
            
            # Append the current shortest distance from city 0 to city n - 1.
            answer.append(dist[n - 1])
        
        return answer

# You can run the following test cases to verify the solution:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    n1 = 5
    queries1 = [[2, 4], [0, 2], [0, 4]]
    print(sol.shortestDistanceAfterQueries(n1, queries1))  # Expected Output: [3, 2, 1]
    
    # Example 2:
    n2 = 4
    queries2 = [[0, 3], [0, 2]]
    print(sol.shortestDistanceAfterQueries(n2, queries2))  # Expected Output: [1, 1]