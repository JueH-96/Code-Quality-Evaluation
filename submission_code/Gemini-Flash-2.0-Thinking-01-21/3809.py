from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)

        # Helper function to calculate the number of distinct common elements
        # Uses sets for efficient intersection
        def intersect(a: List[int], b: List[int]) -> int:
            set_a = set(a)
            set_b = set(b)
            return len(set_a.intersection(set_b))

        # Build the graph using an adjacency list
        # Iterate through all unique pairs of indices (i, j)
        adj = [[] for _ in range(n)]
        for i in range(n):
            # Start j from i + 1 to avoid self-loops (i == j) and duplicate pairs (j, i)
            for j in range(i + 1, n):
                if intersect(properties[i], properties[j]) >= k:
                    # Add edge in both directions for an undirected graph
                    adj[i].append(j)
                    adj[j].append(i)

        # Find the number of connected components using Depth First Search (DFS)
        visited = [False] * n # Keep track of visited nodes
        components = 0 # Counter for connected components

        # DFS helper function to traverse a component
        def dfs(u: int):
            visited[u] = True # Mark the current node as visited
            # Explore all neighbors of the current node
            for v in adj[u]:
                if not visited[v]: # If the neighbor hasn't been visited
                    dfs(v) # Recursively call DFS on the neighbor

        # Iterate through all nodes in the graph
        for i in range(n):
            # If a node hasn't been visited yet, it means it belongs to a new connected component
            if not visited[i]:
                components += 1 # Increment the count of connected components
                dfs(i) # Start a DFS traversal from this node to visit its entire component

        return components