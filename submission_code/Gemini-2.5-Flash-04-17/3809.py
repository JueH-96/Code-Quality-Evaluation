from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)

        # Helper function to calculate the number of distinct integers common to two lists
        def intersect_count(a: List[int], b: List[int]) -> int:
            set_a = set(a)
            set_b = set(b)
            # The intersection of two sets contains elements common to both
            # len() gives the count of these distinct common elements
            return len(set_a.intersection(set_b))

        # Build the graph using an adjacency list
        # adj[i] will store a list of indices j such that there is an edge between i and j
        adj = [[] for _ in range(n)]

        # Iterate through all unique pairs of indices (i, j) where i != j
        # Since the graph is undirected, we only need to check each pair once, e.g., i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check the condition for an edge: intersect(properties[i], properties[j]) >= k
                if intersect_count(properties[i], properties[j]) >= k:
                    # Add edges in both directions for an undirected graph
                    adj[i].append(j)
                    adj[j].append(i)

        # Find the number of connected components using Depth First Search (DFS)
        # Keep track of visited nodes to avoid reprocessing and infinite loops
        visited = [False] * n
        component_count = 0

        # DFS function to traverse a connected component
        def dfs(node):
            visited[node] = True
            # Explore all neighbors of the current node
            for neighbor in adj[node]:
                # If a neighbor hasn't been visited, recurse on it
                if not visited[neighbor]:
                    dfs(neighbor)

        # Iterate through each node
        for i in range(n):
            # If the node has not been visited, it means it belongs to a new, undiscovered component
            if not visited[i]:
                # Increment the component count
                component_count += 1
                # Start a DFS traversal from this node to mark all nodes in its component as visited
                dfs(i)

        # After checking all nodes, component_count holds the total number of connected components
        return component_count