import collections
from typing import List

class Solution:
    """
    Solves the problem of finding the shortest path length from city 0 to city n-1
    after processing a series of queries that add new unidirectional roads.
    Uses Breadth-First Search (BFS) after each query to compute the shortest path length.
    """
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        """
        Calculates the shortest path length from city 0 to city n-1 after each query adds a new road.
        Uses Breadth-First Search (BFS) after each query addition.

        Args:
            n: The number of cities, numbered from 0 to n-1.
            queries: A list of queries, where each query [u, v] represents adding a road from city u to city v.

        Returns:
            A list where the i-th element is the shortest path length from city 0 to city n-1 after processing the first i+1 queries.
        """

        # Initialize adjacency list representation of the graph using collections.defaultdict(list).
        # This structure allows easy addition of neighbors to nodes.
        adj = collections.defaultdict(list)
        
        # Add the initial set of roads: there is a unidirectional road from city i to city i+1
        # for all 0 <= i < n-1. This forms an initial path 0 -> 1 -> ... -> n-1.
        for i in range(n - 1):
            adj[i].append(i + 1)
            
        # List to store the shortest path length calculated after each query.
        answer = []
        
        # Process each query one by one.
        for u, v in queries:
            # Each query [u, v] represents adding a new unidirectional road from city u to city v.
            # Update the graph state by adding this edge to the adjacency list.
            # The problem constraints guarantee 0 <= u < v < n and v - u > 1, and that roads are unique.
            # So we don't need to check for duplicates or invalid edges.
            adj[u].append(v)
            
            # After updating the graph with the new edge, perform Breadth-First Search (BFS)
            # starting from the source city 0 to find the length of the shortest path to the target city n-1.
            
            # Initialize distance array `dist`. dist[i] will store the shortest distance found so far from city 0 to city i.
            # Initialize all distances to -1, signifying that nodes are initially unvisited/unreachable except the source.
            dist = [-1] * n
            # The distance from the source city 0 to itself is 0.
            dist[0] = 0
            
            # Initialize a queue for the BFS algorithm using collections.deque for efficient queue operations (append and popleft).
            # Add the source city 0 to the queue to begin the search.
            queue = collections.deque([0])
            
            # This variable will store the shortest path length once the target node n-1 is reached.
            # Initialize to -1. It's guaranteed to be updated because city n-1 is always reachable from city 0 due to the initial path.
            shortest_path_len = -1 
            
            # Begin the BFS loop. It continues as long as there are nodes in the queue to process.
            while queue:
                # Remove the node at the front of the queue (FIFO). This is the current node to explore.
                curr = queue.popleft()
                
                # Check if the current node is the target city (n-1).
                if curr == n - 1:
                    # If yes, we have found the shortest path to n-1 because BFS explores nodes layer by layer.
                    # Store the length of this shortest path.
                    shortest_path_len = dist[curr]
                    # Optimization: Since BFS finds the shortest path first in unweighted graphs,
                    # we can stop the search early using 'break'. No need to explore further.
                    break 
                
                # Explore all neighbors of the current node `curr`.
                for neighbor in adj[curr]:
                    # Check if the neighbor has been visited yet (its distance is still -1).
                    if dist[neighbor] == -1:
                        # If not visited, calculate its shortest distance from the source. It's one more than the distance to `curr`.
                        dist[neighbor] = dist[curr] + 1
                        # Add the neighbor to the queue to visit its neighbors later in the BFS process.
                        queue.append(neighbor)

            # After the BFS completes (either by finding n-1 and breaking, or by exploring all reachable nodes),
            # `shortest_path_len` contains the length of the shortest path from 0 to n-1 for the current state of the graph.
            # Append this length to the `answer` list.
            answer.append(shortest_path_len)

        # After processing all queries, return the list containing the shortest path lengths computed after each query.
        return answer