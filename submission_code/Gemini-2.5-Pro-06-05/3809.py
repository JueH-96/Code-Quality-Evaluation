from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # If there are 0 or 1 properties, there are n components.
        if n <= 1:
            return n
        
        # --- Union-Find (Disjoint Set Union) Data Structure ---
        # parent[i] stores the parent of item i. Initially, each item is its own parent.
        parent = list(range(n))
        # The number of connected components starts at n. It decreases with each successful union.
        num_components = n

        def find(i: int) -> int:
            """Finds the representative of the set containing element i with path compression."""
            if parent[i] == i:
                return i
            # Path compression: Set the parent of i directly to the root of the set.
            # This flattens the tree, making future finds faster.
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            """Merges the sets containing elements i and j if they are not already in the same set."""
            nonlocal num_components
            root_i = find(i)
            root_j = find(j)
            
            # If the roots are different, the elements are in different sets.
            if root_i != root_j:
                # Merge the two sets by making one root the parent of the other.
                parent[root_j] = root_i
                # Since two components have merged, decrement the total count.
                num_components -= 1
        # --- End of Union-Find ---

        # The problem defines intersection based on *distinct* integers.
        # Converting each property list to a set is the correct and efficient way to handle this.
        # This is done once as a pre-processing step.
        property_sets = [set(p) for p in properties]

        # Iterate through all unique pairs of nodes (properties) to check for edges.
        # The graph is built implicitly by connecting components as we find edges.
        for i in range(n):
            for j in range(i + 1, n):
                # An edge exists if intersect(properties[i], properties[j]) >= k.
                # We calculate the size of the intersection of the two sets.
                # The '&' operator on sets is an efficient way to compute their intersection.
                intersection_size = len(property_sets[i] & property_sets[j])
                
                # If an edge exists, union the components these two nodes belong to.
                if intersection_size >= k:
                    union(i, j)
                    
        return num_components