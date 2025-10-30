import math
from typing import List

# DSU Class Implementation (with iterative find and union by size)
class DSU:
    """
    A Disjoint Set Union (DSU) class, also known as Union-Find, 
    with path compression and union by size optimizations.
    It operates on a set of initial indices.
    """
    def __init__(self, indices):
        """
        Initializes the DSU structure.
        Each index starts as a separate component.
        
        Args:
            indices: An iterable of initial element identifiers (e.g., original indices).
        """
        # Initialize parent pointers to self (each element is its own root)
        self.parent = {i: i for i in indices}
        # Initialize the number of disjoint components
        self.num_components = len(indices)
        # Initialize the size of each component (starts at 1) for union by size
        self.size = {i: 1 for i in indices} 

    def find(self, i):
        """
        Finds the representative (root) of the component containing element i.
        Uses path compression to flatten the tree structure for future efficiency.
        
        Args:
            i: The element index to find the root for.
            
        Returns:
            The representative (root) index of the component.
        
        Raises:
            ValueError: If the index i is not found in the DSU structure.
        """
        # Check if the index exists in the structure. By design in the main
        # function, this should always be the case for called indices.
        if i not in self.parent:
             raise ValueError(f"Index {i} not found in DSU structure.")
             
        root = i
        # Traverse up the tree to find the absolute root
        while self.parent[root] != root:
            root = self.parent[root]
        
        # Path compression: Point all nodes on the path from i to the root
        # directly to the root. This flattens the tree.
        curr = i
        while curr != root: 
            next_node = self.parent[curr] # Remember the next node in the original path
            self.parent[curr] = root      # Point current node directly to the root
            curr = next_node              # Move to the next node in the original path
        return root

    def union(self, i, j):
        """
        Merges the components containing elements i and j.
        Uses union by size: attaches the smaller tree to the root of the larger tree.
        
        Args:
            i: An element in the first component.
            j: An element in the second component.
            
        Returns:
            True if a union was performed (i and j were in different components),
            False otherwise.
        """
        # Find the roots of the components containing i and j
        root_i = self.find(i)
        root_j = self.find(j)
        
        # If they are already in the same component, do nothing
        if root_i != root_j:
            # Union by size heuristic: attach the smaller tree to the larger tree
            # to keep the tree depth relatively small.
            if self.size[root_i] < self.size[root_j]:
                # Swap roots so that root_i always points to the larger tree's root
                root_i, root_j = root_j, root_i 
            
            # Attach the smaller tree (root_j) to the larger tree (root_i)
            self.parent[root_j] = root_i
            # Update the size of the merged component's root (root_i)
            self.size[root_i] += self.size[root_j]
            # Decrease the total number of distinct components by one
            self.num_components -= 1
            return True # Union was performed
        return False # No union needed as they were already connected


class Solution:
    """
    Calculates the number of connected components in a graph defined by an array
    of numbers and a threshold. Two nodes i and j are connected if 
    lcm(nums[i], nums[j]) <= threshold.
    
    The Least Common Multiple (LCM) of a and b is related to the Greatest Common
    Divisor (GCD) by: lcm(a, b) * gcd(a, b) = a * b.
    
    The core idea is to use a Disjoint Set Union (DSU) data structure to group
    indices (nodes) into connected components based on the lcm condition.
    
    We observe that if lcm(a, b) <= threshold, then both a and b must be <= threshold.
    Numbers > threshold form singleton components.
    
    For numbers a, b <= threshold:
    1. If gcd(a, b) = g > 1, they share a common factor. Since lcm(a, b) <= threshold
       implies g <= threshold, we can unite components based on shared factors f > 1
       where f <= threshold.
    2. If gcd(a, b) = 1, the condition becomes a * b <= threshold. These pairs must
       also be united.
    
    The algorithm proceeds in two main steps after handling numbers > threshold and
    the special case where 1 is present.
    """
    def countComponents(self, nums: List[int], threshold: int) -> int:
        
        n = len(nums)
        # Count numbers strictly greater than the threshold. These form singleton components.
        large_count = 0
        # Map relevant values (value <= threshold) to their original index in nums.
        # This map stores numbers that could potentially connect with others.
        relevant_nums_map = {} 
        # Flag to check if the number 1 is present among relevant numbers.
        has_one = False
        
        # First pass: Partition numbers into > threshold and <= threshold.
        # Populate the map for relevant numbers and count large numbers.
        for i in range(n):
            val = nums[i]
            if val > threshold:
                large_count += 1
            else:
                # This number is relevant (value <= threshold)
                relevant_nums_map[val] = i
                if val == 1:
                    # Track the presence of 1, as it connects to all other relevant numbers.
                    has_one = True 

        # If there are no numbers <= threshold, all components are singletons > threshold.
        if not relevant_nums_map:
             return large_count
             
        # Optimization: If 1 is present in nums and <= threshold, all relevant numbers
        # (those <= threshold) will form a single connected component. This is because 
        # lcm(x, 1) = x, and for all relevant x, x <= threshold is true.
        # Thus, 1 connects to all other relevant numbers.
        if has_one:
            return large_count + 1

        # --- Process relevant numbers (all are > 1 and <= threshold at this point) ---
        
        # Get the original indices corresponding to the relevant numbers.
        # These indices will be the elements managed by the DSU structure.
        relevant_indices = list(relevant_nums_map.values())
        # Create a set of the relevant values for efficient O(1) checking (is value present?).
        present_relevant_vals = set(relevant_nums_map.keys()) 
        
        # Initialize Disjoint Set Union (DSU) structure for the relevant indices.
        dsu = DSU(relevant_indices)

        # Step 1: Unite components based on shared common factors > 1.
        # If two relevant numbers a, b share a common factor f (where 1 < f <= threshold),
        # they might be connected (if lcm(a,b) <= threshold). This step proactively unites
        # numbers based on shared factors, covering the case where gcd(a,b) > 1.
        # Iterate through potential factors 'f' from 2 up to the threshold.
        for f in range(2, threshold + 1):
            first_idx_in_multiple_chain = -1
            # Iterate through multiples of 'f' (m_val = k * f) up to the threshold.
            for m_val in range(f, threshold + 1, f):
                # Check if this multiple exists in our set of relevant numbers.
                if m_val in present_relevant_vals:
                    current_idx = relevant_nums_map[m_val]
                    # If this is the first multiple of 'f' encountered in the relevant set for this f.
                    if first_idx_in_multiple_chain == -1:
                        first_idx_in_multiple_chain = current_idx
                    else:
                        # Unite the component of the current multiple with the first one found for this f.
                        # This ensures all multiples of f present in the relevant set belong to the same component.
                        dsu.union(first_idx_in_multiple_chain, current_idx)

        # Step 2: Unite components for pairs (a, b) where gcd(a, b) = 1 and a * b <= threshold.
        # This handles the connections where lcm(a, b) <= threshold but the numbers are coprime.
        # The condition lcm(a, b) <= threshold simplifies to a * b <= threshold when gcd(a, b) = 1.
        # These connections were not necessarily formed in Step 1.
        
        # Get the list of unique relevant values (all are > 1 at this point).
        relevant_vals_list = list(present_relevant_vals) 
        
        # Iterate through each relevant number 'a'.
        for a in relevant_vals_list:
            idx_a = relevant_nums_map[a]
            # Determine the upper limit for potential 'b' values such that a * b <= threshold.
            limit = threshold // a
            # Iterate through potential 'b' values (b > 1) up to the calculated limit.
            # We start b from 2 because the case with 1 was handled, and all relevant_vals > 1.
            for b_val in range(2, limit + 1):
                 # Check if this potential 'b' value is actually present in our relevant numbers.
                 if b_val in present_relevant_vals:
                      # Get the index corresponding to b_val.
                      idx_b = relevant_nums_map[b_val]
                      
                      # Optimization: If 'a' and 'b' are already connected (e.g., via Step 1,
                      # or previous unions in Step 2), skip the potentially expensive gcd check.
                      if dsu.find(idx_a) == dsu.find(idx_b):
                          continue
                          
                      # Check the gcd condition: only unite if gcd(a, b) is 1.
                      # The product condition (a * b <= threshold) is implicitly satisfied by the loop limit.
                      if math.gcd(a, b_val) == 1:
                         # If they are coprime and their product is within the threshold,
                         # unite the components containing 'a' and 'b'.
                         dsu.union(idx_a, idx_b)
                         
        # The final result is the sum of:
        # 1. The count of numbers strictly greater than the threshold (large_count).
        # 2. The number of distinct connected components remaining in the DSU structure 
        #    after processing all relevant numbers and connection conditions.
        return large_count + dsu.num_components