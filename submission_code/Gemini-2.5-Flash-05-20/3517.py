import collections
from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Step 1: Initialize the graph using an adjacency list.
        # adj[i] will store a list of cities reachable directly from city i.
        adj = [[] for _ in range(n)]
        
        # Add the initial unidirectional roads from city i to city i + 1.
        # These roads form a path 0 -> 1 -> 2 -> ... -> n-1.
        for i in range(n - 1):
            adj[i].append(i + 1)
            
        # This list will store the shortest path lengths after each query.
        results = []
        
        # Step 2: Process each query one by one.
        for u, v in queries:
            # Add the new unidirectional road specified by the query.
            adj[u].append(v)
            
            # Step 3: Perform a Breadth-First Search (BFS) to find the shortest
            # path from city 0 to city n-1 in the current graph state.
            
            # The dist array stores the shortest distance from city 0 to each city.
            # Initialize with -1 to indicate unvisited cities.
            dist = [-1] * n
            
            # Create a queue for BFS.
            q = collections.deque()
            
            # Start BFS from city 0.
            dist[0] = 0  # Distance from city 0 to itself is 0.
            q.append(0)  # Add city 0 to the queue.
            
            # Run BFS until the queue is empty or n-1 is reached.
            while q:
                curr_city = q.popleft()
                
                # If we have reached the target city (n-1), we've found the
                # shortest path. We can break early because BFS guarantees
                # the first time a node is visited, it's via the shortest path.
                if curr_city == n - 1:
                    break 
                
                # Explore neighbors of the current city.
                for neighbor in adj[curr_city]:
                    # If the neighbor has not been visited yet, update its distance
                    # and add it to the queue.
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[curr_city] + 1
                        q.append(neighbor)
            
            # After BFS completes (or breaks early), dist[n-1] will hold the
            # length of the shortest path from 0 to n-1.
            # The initial path 0->1->...->n-1 always exists and ensures n-1 is reachable.
            results.append(dist[n - 1])
            
        return results