import collections
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize adjacency list with initial roads
        # Roads from i to i + 1 for 0 <= i < n - 1
        # Using a list of lists to represent the adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
            
        answer = []
        
        # Process each query
        for u, v in queries:
            # Add the new unidirectional road from u to v
            # The constraint 0 <= u < v < n and 1 < v - u ensures this is a valid forward edge
            adj[u].append(v)
            
            # Perform BFS starting from city 0 to find the shortest distance to city n-1
            # dist[i] will store the shortest distance from 0 to i
            dist = [-1] * n # Initialize distances. -1 means unvisited
            dist[0] = 0    # Distance from source (0) to itself is 0
            
            # Queue for BFS. Stores nodes to visit.
            queue = collections.deque([0]) 
            
            # Run BFS
            while queue:
                curr = queue.popleft()
                
                # If we reached the target city (n-1), we found the shortest path.
                # Since BFS explores layer by layer in an unweighted graph, the first time
                # we reach the target, it's via the shortest path in terms of number of edges.
                if curr == n - 1:
                    # The shortest distance is dist[n-1]. No need to continue BFS.
                    break 
                    
                # Explore neighbors of the current city
                for neighbor in adj[curr]:
                    # If the neighbor has not been visited yet (distance is -1)
                    if dist[neighbor] == -1:
                        # Mark neighbor as visited and set its distance
                        # The distance to the neighbor is 1 more than the distance to the current node
                        dist[neighbor] = dist[curr] + 1
                        # Add neighbor to the queue for exploration
                        queue.append(neighbor)
            
            # After BFS completes (either by reaching n-1 or exhausting the queue),
            # dist[n-1] contains the length of the shortest path from 0 to n-1.
            # Since the initial graph has a path 0->...->n-1 and only forward edges
            # are added, n-1 is always reachable from 0.
            answer.append(dist[n - 1])
            
        return answer