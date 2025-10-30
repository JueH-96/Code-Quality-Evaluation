from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)

        # Helper function to find the number of distinct common integers
        # This function takes two lists of integers and returns the count
        # of distinct integers that are present in both lists.
        def intersect(arr1: List[int], arr2: List[int]) -> int:
            # Convert lists to sets for efficient intersection checking.
            # Sets store only unique elements, handling duplicates within arr1/arr2 automatically.
            set1 = set(arr1)
            set2 = set(arr2)
            
            # The intersection of two sets contains elements common to both.
            # The length of this intersection set is the number of distinct common integers.
            return len(set1.intersection(set2))

        # Build the adjacency list for the graph.
        # adj[i] will store a list of nodes j that have an edge with node i.
        adj = [[] for _ in range(n)]
        
        # Iterate through all unique pairs of nodes (i, j) where i < j.
        # This avoids redundant checks (e.g., checking (j, i) if (i, j) is already checked)
        # and self-loops (i, i).
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the intersection condition is met to form an edge.
                # An edge exists if the number of common distinct integers is >= k.
                if intersect(properties[i], properties[j]) >= k:
                    # If the condition is met, add an edge between i and j.
                    # Since the graph is undirected, add j to i's adjacency list
                    # and i to j's adjacency list.
                    adj[i].append(j)
                    adj[j].append(i)

        # Count connected components using Depth First Search (DFS).
        # visited array keeps track of nodes that have already been visited
        # during a traversal to avoid reprocessing and infinite loops.
        visited = [False] * n
        num_components = 0 # Counter for the number of connected components

        # Iterate through each node in the graph.
        for i in range(n):
            # If the current node 'i' has not been visited yet, it means
            # it belongs to a new connected component.
            if not visited[i]:
                num_components += 1 # Increment the component count
                
                # Start a DFS traversal from node 'i' to find all nodes
                # in this new connected component.
                stack = [i] # Use a stack for iterative DFS
                visited[i] = True # Mark the starting node as visited
                
                while stack:
                    u = stack.pop() # Get the current node from the stack
                    
                    # Explore all neighbors of the current node 'u'.
                    for v in adj[u]:
                        # If a neighbor 'v' has not been visited, it means
                        # it's part of the current component.
                        if not visited[v]:
                            visited[v] = True # Mark it as visited
                            stack.append(v) # Add it to the stack for further exploration
                            
        # After iterating through all nodes, num_components will hold
        # the total number of connected components.
        return num_components