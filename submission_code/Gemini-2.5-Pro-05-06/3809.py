import collections
from typing import List

class Solution:
    def _intersect(self, arr1: List[int], arr2: List[int]) -> int:
        """
        Helper function to calculate the number of distinct common integers
        between two lists.
        """
        set1 = set(arr1)
        set2 = set(arr2)
        
        # The intersection of two sets gives a new set containing elements
        # present in both original sets.
        common_elements = set1.intersection(set2)
        return len(common_elements)

    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)

        # Constraints state n >= 1. If n could be 0, we might add:
        # if n == 0:
        #     return 0
        
        # Adjacency list to represent the graph
        # adj[i] will store a list of neighbors of node i
        adj = [[] for _ in range(n)]

        # Construct the graph:
        # Iterate over all unique pairs of nodes (i, j)
        for i in range(n):
            # j starts from i + 1 to consider each pair once and avoid self-loops
            for j in range(i + 1, n): 
                # Calculate the number of distinct common integers
                num_common = self._intersect(properties[i], properties[j])
                
                # If the number of common integers is at least k, add an undirected edge
                if num_common >= k:
                    adj[i].append(j)
                    adj[j].append(i)

        # Count connected components using BFS
        visited = [False] * n  # Keep track of visited nodes
        num_components = 0

        for i in range(n):
            if not visited[i]:
                # This node i is part of a new, unvisited connected component
                num_components += 1
                
                # Perform BFS starting from node i to find all nodes in this component
                q = collections.deque([i]) # Initialize queue with the starting node
                visited[i] = True
                
                while q:
                    u = q.popleft()  # Current node being processed
                    
                    # Visit all unvisited neighbors
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
        
        return num_components