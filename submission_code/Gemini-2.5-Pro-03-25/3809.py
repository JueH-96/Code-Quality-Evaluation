import collections # Not strictly needed for the DSU approach, but good practice if List was ambiguous.
from typing import List # Required for type hinting

# --- DSU Class Definition ---
# It's generally cleaner to define helper classes like DSU outside the main Solution class.
class DSU:
    """
    Disjoint Set Union (Union-Find) data structure.
    Supports path compression and union by size for optimized performance.
    Used here to efficiently track and count connected components in the graph.
    """
    def __init__(self, n: int):
        """
        Initializes the DSU structure for n elements (nodes).
        Each element starts in its own component.
        """
        # parent[i] stores the parent of element i in the tree representation.
        # Initially, each element is its own parent (root of its own component).
        self.parent = list(range(n))
        # size[i] stores the size (number of nodes) of the component rooted at i.
        # Used for the union-by-size optimization. Initially, each component has size 1.
        self.size = [1] * n
        # num_components tracks the total number of disjoint sets (connected components).
        # Starts at n and decreases as components are merged.
        self.num_components = n

    def find(self, i: int) -> int:
        """
        Finds the representative (root) of the set containing element i.
        Implements path compression: during the find operation, updates the parent
        pointers of nodes along the path to point directly to the root. This flattens
        the tree structure and speeds up future find operations.
        """
        # 1. Find the root of the component containing i.
        # The root is an element whose parent is itself.
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        
        # 2. Path compression: Make all nodes on the path from i to the root
        # point directly to the root.
        curr = i
        while self.parent[curr] != root:
            # Temporarily store the original parent of the current node.
            next_node = self.parent[curr]
            # Set the parent of the current node directly to the root.
            self.parent[curr] = root
            # Move to the next node up the path.
            curr = next_node
            
        return root

    def union(self, i: int, j: int) -> bool:
        """
        Merges the sets containing elements i and j if they are not already in the same set.
        Implements union by size: attaches the root of the smaller component tree
        to the root of the larger component tree. This helps keep the trees balanced
        and the find operation efficient.

        Args:
            i: The first element.
            j: The second element.

        Returns:
            True if i and j were in different sets and a merge occurred, False otherwise.
        """
        # Find the roots of the components containing i and j.
        root_i = self.find(i)
        root_j = self.find(j)
        
        # If i and j are already in the same component (same root), do nothing.
        if root_i != root_j:
            # Union by size: Attach the smaller tree to the root of the larger tree.
            if self.size[root_i] < self.size[root_j]:
                # Make root_j the parent of root_i.
                self.parent[root_i] = root_j
                # Update the size of the merged component rooted at root_j.
                self.size[root_j] += self.size[root_i]
            else: # If sizes are equal, or size[root_i] >= size[root_j]
                # Make root_i the parent of root_j.
                self.parent[root_j] = root_i
                # Update the size of the merged component rooted at root_i.
                self.size[root_i] += self.size[root_j]
            
            # Since two components were merged, decrement the total number of components.
            self.num_components -= 1
            # Return True to indicate that a merge happened.
            return True 
            
        # If roots were the same, no merge occurred.
        return False

# --- Solution Class ---
# Using the provided starter code structure
class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        """
        Calculates the number of connected components in an undirected graph
        constructed based on the intersections of property lists.

        The graph has n nodes, where n is the number of property lists (len(properties)).
        Node i corresponds to properties[i].
        An edge exists between node i and node j (i != j) if and only if the number
        of distinct integers common to properties[i] and properties[j] is at least k.

        Args:
            properties: A list of lists of integers. properties[i] contains the
                        properties associated with node i.
            k: The minimum number of common properties required to form an edge.

        Returns:
            The number of connected components in the constructed graph.
        """
        n = len(properties)
        # Handle the edge case of an empty input list.
        if n == 0:
            return 0

        # Initialize the Disjoint Set Union (DSU) data structure with n elements.
        # Each element initially represents a separate component.
        dsu = DSU(n)

        # Precompute the set representation of each property list.
        # Converting lists to sets allows for efficient intersection calculation (O(min(len(A), len(B))) on average).
        # Doing this outside the loops avoids repeated conversions.
        property_sets = [set(p) for p in properties]

        # Iterate through all unique pairs of nodes (properties) (i, j) where i < j.
        for i in range(n):
            set_i = property_sets[i]
            # The inner loop starts from i + 1 to consider each pair only once (avoids (j, i))
            # and avoids comparing a node with itself (avoids (i, i)).
            for j in range(i + 1, n):
                set_j = property_sets[j]

                # Calculate the number of distinct integers common to both sets.
                # The intersection method of sets returns a new set with common elements.
                # The length of this intersection set gives the required count.
                intersection_count = len(set_i.intersection(set_j))

                # Check if the intersection count meets the threshold k.
                # If it does, an edge exists between nodes i and j.
                if intersection_count >= k:
                    # Perform the union operation in the DSU structure.
                    # If nodes i and j are in different components, this merges them
                    # and decrements the total component count within the DSU object.
                    dsu.union(i, j)

        # After iterating through all relevant pairs and performing unions for edges,
        # the `num_components` attribute of the DSU object holds the final count
        # of connected components in the graph.
        return dsu.num_components