from typing import List

class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        
        # Adjacency list to represent the graph.
        # Using a list of lists is efficient as city numbers are 0 to n-1.
        adj = [[] for _ in range(n)]
        
        # Initially, there is a unidirectional road from city i to city i + 1.
        for i in range(n - 1):
            adj[i].append(i + 1)
        
        answer = []
        
        # Process each query sequentially.
        for u, v in queries:
            # Add the new road to the graph.
            adj[u].append(v)
            
            # We need to find the shortest path in an unweighted Directed Acyclic Graph (DAG).
            # The constraints (u < v for all edges) ensure the graph is a DAG, and
            # the nodes 0, 1, ..., n-1 are in a valid topological order.
            # We can use dynamic programming over this topological sort to find the shortest paths
            # from the source node 0. This is equivalent to running BFS on a DAG.
            
            # `dist[i]` will store the length of the shortest path from city 0 to city i.
            # Initialize distances to infinity, except for the source city 0.
            dist = [float('inf')] * n
            dist[0] = 0
            
            # Iterate through each node in topological order.
            for i in range(n):
                # For each neighbor of the current node, we "relax" the edge,
                # potentially finding a shorter path to the neighbor.
                # The distance to a neighbor is the minimum of its current known distance
                # and the distance of a path passing through the current node `i`.
                for neighbor in adj[i]:
                    dist[neighbor] = min(dist[neighbor], dist[i] + 1)
            
            # After iterating through all nodes, dist[n-1] holds the shortest distance.
            answer.append(dist[n-1])
            
        return answer